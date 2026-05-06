"""
Shared background scoring for all speaking practice types.
Downloads audio from S3, runs Azure scoring, applies rubric weights + PTE formula.
Ported from englishfirm-app-fastapi/services/question_service.py _score_free_form_bg.
"""
import logging
import re
import threading
import requests as _requests

import core.config as config

logger = logging.getLogger(__name__)
from services.s3_service import generate_presigned_url
from services.session_service import store_score, update_speaking_score_in_db
from services.scoring.azure_scorer import _compute_question_score

from core.logging_config import get_logger

log = get_logger(__name__)


def _pte_score(pct: float) -> int:
    return max(config.PTE_FLOOR, min(config.PTE_CEILING, round(config.PTE_BASE + pct * config.PTE_SCALE)))


_FLUENCY_FORMULA_TYPES = {
    'read_aloud',
    'repeat_sentence',
    'describe_image',
    'retell_lecture',
    'respond_to_situation',
    'ptea_respond_situation',
    'summarize_group_discussion',
}


# Pause / silence tuning constants. Bumping pause_min_ms from 300 → 600 means
# only true breaks count as pauses; sub-600 ms gaps are treated as natural
# in-sentence hesitations. Trailing tolerance widened from 100 → 200 ms so
# small mouth-noise/clicks at the recorder cut don't count as a final pause.
_PAUSE_MIN_MS = 600
_LEADING_TOLERANCE_MS = 200
_TRAILING_TOLERANCE_MS = 200
_SILENCE_THRESH_DBFS = -30


def _silence_ratio_pct(seg) -> tuple:
    """
    Return (silence_ratio_pct, pause_count) for the within-speech window.

    silence_ratio_pct: total within-speech silence (>= _PAUSE_MIN_MS) as a
    percentage [0, 100] of the speech window (audio - leading - trailing).
    pause_count: number of within-speech silence segments (excludes leading
    and trailing).

    Uses pydub silence detection at -30 dBFS, min 600 ms.
    """
    try:
        from pydub.silence import detect_silence
    except Exception:
        return 0.0, 0
    total_ms = len(seg)
    if total_ms <= 0:
        return 0.0, 0
    silences = detect_silence(
        seg, min_silence_len=_PAUSE_MIN_MS, silence_thresh=_SILENCE_THRESH_DBFS
    )
    if not silences:
        return 0.0, 0

    leading_ms = 0
    trailing_ms = 0
    within_pauses = []
    for s, e in silences:
        if s <= _LEADING_TOLERANCE_MS and leading_ms == 0:
            leading_ms = e - s
        elif (total_ms - e) <= _TRAILING_TOLERANCE_MS:
            trailing_ms = e - s
        else:
            within_pauses.append((s, e))

    speech_window_ms = total_ms - leading_ms - trailing_ms
    if speech_window_ms <= 0:
        return 0.0, len(within_pauses)
    within_ms = sum(e - s for s, e in within_pauses)
    ratio = max(0.0, min(100.0, within_ms / speech_window_ms * 100.0))
    return ratio, len(within_pauses)


def _wpm_score(wpm: float, question_type: str) -> float:
    """
    Per-type WPM curve. Returns the WPM-side fluency score in [0, 100].

    Non-SGD speaking types (RA, RS, DI, RL, RTS, ptea_RTS):
        floor gate: wpm < 80 → 0
        ascending:  80 → 130   (slope 2.0/wpm)
        plateau:    130 ≤ wpm ≤ 200 → 100
        descending: 200 → 250  (slope 2.0/wpm)
        ceiling gate: wpm > 250 → 0

    SGD only:
        no floor gate (ascend starts at 0)
        ascending:   0 → 70    (slope ≈ 1.43/wpm)
        plateau:    70 ≤ wpm ≤ 200 → 100
        descending: 200 → 250  (slope 2.0/wpm)
        ceiling gate: wpm > 250 → 0

    SGD's wider plateau and missing floor reflect the longer-form,
    slower-paced nature of summarising a multi-speaker discussion.
    Caller is responsible for the WPM gate hard-fail; this function
    returns only the curve value (still 0 at the boundaries).
    """
    if question_type == 'summarize_group_discussion':
        if wpm > 250.0:
            return 0.0
        if wpm < 70.0:
            return max(0.0, 100.0 - (100.0 / 70.0) * (70.0 - wpm))
        if wpm <= 200.0:
            return 100.0
        return max(0.0, 100.0 - 2.0 * (wpm - 200.0))

    # All other speaking types
    if wpm < 80.0 or wpm > 250.0:
        return 0.0
    if wpm < 130.0:
        return 100.0 - 2.0 * (130.0 - wpm)
    if wpm <= 200.0:
        return 100.0
    return 100.0 - 2.0 * (wpm - 200.0)


def _cross_multiplier(score: float) -> float:
    """
    Soft cross-penalty multiplier. Returns 1.0 when the dimension is healthy
    (>= 20), 0.5 when it bottomed out (0), and a linear interpolation in
    between. This replaces the rev-13 binary halving (fluency==0 → halve).

    The full-strength penalty zone is 0–20; any score >= 20 contributes no
    penalty to the other dimensions.
    """
    if score >= 20.0:
        return 1.0
    if score <= 0.0:
        return 0.5
    return 0.5 + 0.025 * score   # equivalently 0.5 + 0.5 * (score / 20)


def _count_sentences(text: str) -> int:
    """
    Count sentences via terminal punctuation. Used to gate the silence rule:
    one pause per sentence boundary is natural, so silence only penalises
    fluency when pause_count > sentence_count.

    For RA / RS this is the reference passage; for DI / RL / RTS / SGD / ASQ
    this is the user's transcript (Whisper or Azure).
    Minimum 1 — never gate against zero.
    """
    if not text:
        return 1
    # Split on . ! ? — treat each non-empty fragment as a sentence.
    parts = [p.strip() for p in re.split(r'[.!?]+', text) if p.strip()]
    return max(1, len(parts))


def _apply_speaking_fluency_formula(
    user_id: int,
    question_id: int,
    question_type: str,
    audio_bytes: bytes,
    word_scores: list,
    content: float,
    fluency: float,
    pronunciation: float,
    reference_text: str = "",
    transcript: str = "",
):
    """
    Replace Azure's FluencyScore with a deterministic formula based on WPM
    and within-speech silence. Content (CompletenessScore) and pronunciation
    (AccuracyScore) pass through Azure-as-is.

    Pauses are silence segments >= 600 ms within the speech window. The
    silence side of the formula only fires when pause_count > sentence_count
    (one natural pause per sentence is not penalised — only excess hesitation
    is). Sentence count: reference passage for RA/RS, user transcript for the
    free-form types.

    Formula:
        sentences  = sentence count (passage for RA/RS, transcript otherwise)
        pauses     = within-speech silence segments >= 600 ms
        sil_rule   = pauses > sentences

        if wpm < 100  OR  wpm > 240:
            fluency = 0
        elif sil_rule and silence_pct > 20:
            fluency = 0
        else:
            wpm_score = 100 - 2.5 * (140 - wpm)   if 100 <= wpm < 140
                        100                        if 140 <= wpm <= 200
                        100 - 2.5 * (wpm - 200)    if 200 < wpm <= 240
            if sil_rule:
                silence_score = 5 * (20 - silence_pct)
                fluency       = min(wpm_score, silence_score)
            else:
                fluency       = wpm_score   # silence side ignored

    WPM curve: 100 floor → ascending to 100 at 140, plateau through 200,
    descending to 0 at 240, ceiling above. Symmetric ±2.5/WPM slopes.

    Words counted for WPM: word_scores entries where error_type is neither
    'Omission' nor 'Insertion'. Same rule as before.

    Coverage: types in _FLUENCY_FORMULA_TYPES, across practice + sectional + mock.
    answer_short_question is excluded.

    Fail-open: any error keeps Azure's original c/f/p (logged).
    """
    if question_type not in _FLUENCY_FORMULA_TYPES:
        return content, fluency, pronunciation, {}
    try:
        from pydub import AudioSegment
        import io

        words_spoken = sum(
            1 for w in (word_scores or [])
            if isinstance(w, dict) and w.get('error_type') not in ('Omission', 'Insertion')
        )
        if words_spoken <= 0:
            return content, fluency, pronunciation, {}

        seg = AudioSegment.from_file(io.BytesIO(audio_bytes))
        duration_sec = seg.duration_seconds
        if duration_sec <= 0:
            return content, fluency, pronunciation, {}

        wpm = words_spoken * 60.0 / duration_sec
        silence_pct, pause_count = _silence_ratio_pct(seg)

        # Sentence count — reference passage for RA/RS, transcript otherwise.
        if question_type in ('read_aloud', 'repeat_sentence'):
            sentence_count = _count_sentences(reference_text)
        else:
            sentence_count = _count_sentences(transcript)

        silence_rule_applies = pause_count > sentence_count

        # Per-type gates and silence threshold. SGD (longer-form discussion
        # summaries) gets a wider plateau on the WPM curve, no floor gate,
        # and a much higher silence kill threshold (75% vs 20%).
        is_sgd = question_type == 'summarize_group_discussion'
        silence_kill_threshold = 75.0 if is_sgd else 20.0

        # WPM gate (hard fail)
        if (is_sgd and wpm > 250.0) or (not is_sgd and (wpm < 80.0 or wpm > 250.0)):
            new_fluency = 0.0
            wpm_score_str = "-"
            sil_score_str = "-"
            reason_parts = []
            if not is_sgd and wpm < 80.0:
                reason_parts.append(f"wpm<80({wpm:.1f})")
            if wpm > 250.0:
                reason_parts.append(f"wpm>250({wpm:.1f})")
            reason = ",".join(reason_parts)

        # Silence kill switch (per-type threshold, gated by sentence count)
        elif silence_rule_applies and silence_pct > silence_kill_threshold:
            new_fluency = 0.0
            wpm_score_str = "-"
            sil_score_str = "-"
            reason = (
                f"sil>{silence_pct:.1f}%(p{pause_count}>s{sentence_count},"
                f"thresh={int(silence_kill_threshold)}%)"
            )

        # Normal scoring
        else:
            wpm_score = _wpm_score(wpm, question_type)

            if silence_rule_applies:
                # Per-type silence_score formula. SGD's 75-band scales 100→0
                # over silence_pct 0→75 (slope ≈ 1.33/pct). Others use the
                # original 20-band (slope 5/pct).
                if is_sgd:
                    silence_score = max(0.0, 100.0 * (75.0 - silence_pct) / 75.0)
                else:
                    silence_score = max(0.0, 5.0 * (20.0 - silence_pct))
                new_fluency = min(wpm_score, silence_score)
                sil_score_str = f"{silence_score:.1f}"
                reason = "ok"
            else:
                new_fluency = wpm_score
                sil_score_str = f"skip(p{pause_count}<=s{sentence_count})"
                reason = "ok_sil_skipped"
            new_fluency = max(0.0, min(100.0, new_fluency))
            wpm_score_str = f"{wpm_score:.1f}"

        log.info(
            "[FLUENCY_FORMULA] q=%s type=%s user=%s words=%s dur=%.2fs "
            "wpm=%.1f sil=%.1f%% pauses=%d sentences=%d "
            "wpm_score=%s sil_score=%s azure_f=%.1f → new_f=%.1f reason=%s",
            question_id, question_type, user_id, words_spoken, duration_sec,
            wpm, silence_pct, pause_count, sentence_count,
            wpm_score_str, sil_score_str,
            float(fluency), new_fluency, reason,
        )

        fluency_metrics = {
            "wpm": round(wpm, 1),
            "silence_pct": round(silence_pct, 1),
            "pause_count": pause_count,
            "sentence_count": sentence_count,
            "silence_rule_applied": silence_rule_applies,
            "duration_sec": round(duration_sec, 2),
        }

        # Per-type cross-penalty. RA / RS / DI use the full symmetric
        # ramp (every dimension drags the other two). RL / RTS /
        # ptea_RTS / SGD isolate content — content is never dragged in
        # by F or P, but content's mC still drags F and P (bad content
        # in a free-form response should pull down delivery credit).
        # F and P don't drag each other in the free-form branch.
        #
        # Content==0 still hits its hard zero-out rule in azure_scorer
        # (_CONTENT_ZERO_TASKS / _CONTENT_ZERO_LLM_TASKS) downstream.
        before_c = float(content)
        before_f = new_fluency
        before_p = float(pronunciation)

        mC = _cross_multiplier(before_c)
        mF = _cross_multiplier(before_f)
        mP = _cross_multiplier(before_p)

        content_drags_others = question_type in {
            'read_aloud', 'repeat_sentence', 'describe_image',
        }

        if content_drags_others:
            # RA / RS / DI: full symmetric (Option B)
            new_content       = before_c * mF * mP    # F and P drag C
            new_fluency       = before_f * mC * mP    # C and P drag F
            new_pronunciation = before_p * mC * mF    # C and F drag P
        else:
            # RL / RTS / ptea_RTS / SGD: content isolated, only mC drags F+P
            new_content       = before_c              # no drag in
            new_fluency       = before_f * mC         # only C drags F
            new_pronunciation = before_p * mC         # only C drags P

        # Log only when at least one dimension is in the penalty zone
        # (otherwise all three multipliers are 1.0 and there is nothing to log).
        if mC < 1.0 or mF < 1.0 or mP < 1.0:
            log.info(
                "[CROSS_PENALTY] q=%s type=%s user=%s mC=%.3f mF=%.3f mP=%.3f "
                "before c/f/p=%.1f/%.1f/%.1f → after c/f/p=%.1f/%.1f/%.1f",
                question_id, question_type, user_id,
                mC, mF, mP,
                before_c, before_f, before_p,
                new_content, new_fluency, new_pronunciation,
            )
            fluency_metrics["cross_multipliers"] = {
                "mC": round(mC, 3),
                "mF": round(mF, 3),
                "mP": round(mP, 3),
            }

        return new_content, new_fluency, new_pronunciation, fluency_metrics
    except Exception as e:
        log.error("[FLUENCY_FORMULA] application failed (fail-open, keeping azure fluency): %s", e)
        return content, fluency, pronunciation, {}


# ── RA v2: Whisper-driven content + pause-based fluency ────────────────────────
#
# Replaces the legacy `score_read_aloud → _apply_speaking_fluency_formula`
# pipeline for read_aloud only. Drops Azure CompletenessScore (replaced by
# positional Whisper match against the reference) and amplitude-based silence
# detection (replaced by Whisper inter-word gap + filler regex).
#
# Pronunciation still comes from Azure AccuracyScore — phoneme-level scoring
# is Azure's strength and Whisper has nothing equivalent.

_HESITATION_RE = re.compile(r"^(?:u+h+|u+m+|a+h+|aa+h+|hm+|m+hm+|er+|erm+)$", re.IGNORECASE)

# RA v3: pause detection switched from Azure inter-word gaps to pydub
# continuous-silence detection. Azure attributes long word-tail decay to
# word duration, so it missed perceptible pauses (e.g. Sekar's 700–1200 ms
# inter-syllable quiets were marked as part of the previous word). Pydub
# at ≥ 500 ms below -30 dBFS is closer to what a human listener perceives
# as a pause. Leading + trailing dead air still excluded with 200 ms
# tolerance.
_RA_PAUSE_MIN_MS         = 500
_RA_PAUSE_LEADING_TOL_MS = 200
_RA_PAUSE_TRAILING_TOL_MS = 200

# How many pauses + hesitations to surface on the score view, longest first.
_RA_BREAKDOWN_CAP = 10

# Common English contractions — expand both directions so "we'll" ≡ "we will".
_CONTRACTIONS = {
    "i'm": "i am", "you're": "you are", "we're": "we are", "they're": "they are",
    "it's": "it is", "he's": "he is", "she's": "she is", "that's": "that is",
    "what's": "what is", "where's": "where is", "there's": "there is",
    "i'll": "i will", "you'll": "you will", "we'll": "we will",
    "they'll": "they will", "he'll": "he will", "she'll": "she will",
    "i've": "i have", "you've": "you have", "we've": "we have",
    "they've": "they have",
    "i'd": "i would", "you'd": "you would", "we'd": "we would",
    "they'd": "they would", "he'd": "he would", "she'd": "she would",
    "isn't": "is not", "aren't": "are not", "wasn't": "was not",
    "weren't": "were not", "hasn't": "has not", "haven't": "have not",
    "hadn't": "had not", "doesn't": "does not", "don't": "do not",
    "didn't": "did not", "won't": "will not", "wouldn't": "would not",
    "shan't": "shall not", "shouldn't": "should not", "can't": "can not",
    "cannot": "can not", "couldn't": "could not", "mustn't": "must not",
    "let's": "let us",
}


def _ra_normalise_tokens(text: str) -> list:
    """Lowercase, expand contractions, strip punctuation, return token list."""
    if not text:
        return []
    t = text.lower().strip()
    # Normalise typographic apostrophes/quotes to ASCII before contraction match
    t = t.replace("’", "'").replace("‘", "'")
    # Expand contractions phrase-by-phrase
    for src, dst in _CONTRACTIONS.items():
        t = re.sub(rf"\b{re.escape(src)}\b", dst, t)
    # Strip everything except letters, digits, apostrophes-in-words, whitespace
    t = re.sub(r"[^\w\s']", " ", t)
    return [tok for tok in t.split() if tok]


def _build_ra_pause_breakdown(
    *,
    pause_intervals: list,
    whisper_words: list,
    azure_words: list,
) -> tuple:
    """
    Build the longest-first list of pauses + hesitation clusters with
    surrounding word context, capped at _RA_BREAKDOWN_CAP. Returns
    (breakdown_list, overflow_count).

    Each entry shape:
      pause:
        {kind: "pause", duration_ms: int,
         preceding_word: str|None, following_word: str|None, count: 1}
      hesitation cluster:
        {kind: "hesitation", text: str, count: int, duration_ms: int,
         preceding_word: str|None, following_word: str|None}

    Word context priority:
      - Pauses: lookup against Azure word offsets (10 ms precision); fall
        back to Whisper word timings if Azure unavailable.
      - Hesitations: lookup against Whisper transcript order (the source
        of truth for "uh"/"um"/etc).
    """
    entries = []

    # Helper: word-by-timestamp lookup for pauses against Azure offsets.
    # Each Azure word: offset_ms, duration_ms, word.
    az_words = list(azure_words or [])
    az_words.sort(key=lambda w: w.get("offset_ms", 0))

    def _word_before_pause(start_ms: int):
        prev = None
        for w in az_words:
            end_ms = w.get("offset_ms", 0) + w.get("duration_ms", 0)
            if end_ms <= start_ms:
                prev = w
            else:
                break
        # assess_pronunciation_with_timestamps emits "Word" (capital W);
        # fall back to lowercase for any alternate source.
        if not prev:
            return None
        return prev.get("Word") or prev.get("word") or None

    def _word_after_pause(end_ms: int):
        for w in az_words:
            if w.get("offset_ms", 0) >= end_ms:
                return w.get("Word") or w.get("word") or None
        return None

    # Pauses — track start_ms for chronological ordering
    for s, e in pause_intervals:
        entries.append({
            "kind":           "pause",
            "duration_ms":    int(e - s),
            "preceding_word": _word_before_pause(int(s)),
            "following_word": _word_after_pause(int(e)),
            "count":          1,
            "_sort_ms":       int(s),
        })

    # Hesitation clustering — walk Whisper words in order, group consecutive
    # identical hesitation tokens into one entry.
    ws = list(whisper_words or [])

    def _normalised(token: str) -> str:
        return token.strip().rstrip(".,!?").lower()

    def _is_hesitation(token: str) -> bool:
        return bool(_HESITATION_RE.match(_normalised(token)))

    i = 0
    while i < len(ws):
        w = ws[i]
        text = _normalised(w.get("text", ""))
        if not _is_hesitation(text):
            i += 1
            continue
        # Cluster consecutive identical hesitations
        j = i + 1
        while j < len(ws) and _normalised(ws[j].get("text", "")) == text:
            j += 1
        cluster = ws[i:j]
        cluster_start = cluster[0].get("start", 0.0)
        cluster_end   = cluster[-1].get("end", cluster_start)
        cluster_dur_ms = max(0, int(round((cluster_end - cluster_start) * 1000)))

        # Surrounding words from Whisper, skipping any other hesitations
        before = None
        for k in range(i - 1, -1, -1):
            tk = _normalised(ws[k].get("text", ""))
            if tk and not _is_hesitation(tk):
                before = ws[k].get("text", "").strip()
                break
        after = None
        for k in range(j, len(ws)):
            tk = _normalised(ws[k].get("text", ""))
            if tk and not _is_hesitation(tk):
                after = ws[k].get("text", "").strip()
                break

        entries.append({
            "kind":           "hesitation",
            "text":           text,
            "count":          len(cluster),
            "duration_ms":    cluster_dur_ms,
            "preceding_word": before,
            "following_word": after,
            "_sort_ms":       int(round(cluster_start * 1000)),
        })
        i = j

    # Chronological order (in order of appearance in the recording).
    # Cap at _RA_BREAKDOWN_CAP — keep the FIRST N (start of the passage),
    # since "what tripped you up first" reads naturally with the passage.
    entries.sort(key=lambda x: x.get("_sort_ms", 0))
    overflow = max(0, len(entries) - _RA_BREAKDOWN_CAP)
    capped = entries[:_RA_BREAKDOWN_CAP]
    # Strip the internal sort key before returning
    for e in capped:
        e.pop("_sort_ms", None)
    return capped, overflow


def _score_read_aloud_v2(
    user_id: int,
    question_id: int,
    audio_bytes: bytes,
    reference_text: str,
) -> dict:
    """
    Read Aloud scoring path using:

      - Whisper word-level transcription for content (positional match) and
        speech-window WPM + pause detection.
      - Azure pronunciation assessment for AccuracyScore (the pronunciation
        component) and per-word phoneme scores for trainer review.

    Returns the same shape `_run_scoring` expects:

      {
        "content":            0-100 (positional match against reference),
        "fluency":            0-100 (after WPM band + pause penalty),
        "pronunciation":      0-100 (Azure AccuracyScore unchanged),
        "transcript":         Whisper transcript,
        "word_scores":        Azure per-word phoneme list,
        "fluency_metrics":    diagnostic dict for the trainer review UI,
      }
    """
    from services.azure_speech_service import assess_pronunciation_with_timestamps
    from services.whisper_service import transcribe_with_whisper_words

    # Azure (continuous + word timestamps) for pronunciation + per-word
    # scores + per-word offsets used for pause detection. Wrapped so a
    # transient Azure failure doesn't kill the whole RA scoring — content
    # can still be computed from Whisper alone.
    azure_words = []          # for pause detection (offset_ms + duration_ms)
    try:
        azure = assess_pronunciation_with_timestamps(audio_bytes, reference_text)
        pronunciation = float(azure.get("AccuracyScore") or 0)
        azure_words = azure.get("Words", []) or []
        word_scores = [
            {
                "word":            w.get("Word", ""),
                "accuracy":        w.get("AccuracyScore", 0),
                "error_type":      w.get("ErrorType", "None"),
            }
            for w in azure_words
        ]
    except Exception as e:
        log.warning("[RA_V2] Azure pronunciation failed (continuing with 0): %s", e)
        pronunciation = 0.0
        word_scores = []

    # Whisper for transcript (used for content matching — Whisper handles
    # accented PTE vocabulary better than Azure on free-speech-like reads).
    whisper = transcribe_with_whisper_words(audio_bytes)
    whisper_words = whisper.get("words", []) or []
    transcript = whisper.get("transcript", "") or ""

    # ── Content score: multiset presence ────────────────────────────────────
    # Each reference token earns a hit if the same token appears anywhere in
    # the spoken transcript (multiset-aware: a reference word repeated twice
    # needs to be spoken twice for both hits). Robust to single token merges
    # like "wild cat" → "wildcat" — only the merged tokens miss, the rest
    # of the passage still scores.
    from collections import Counter as _Counter
    ref_tokens = _ra_normalise_tokens(reference_text)
    spoken_tokens = _ra_normalise_tokens(transcript)
    if ref_tokens:
        spoken_counts = _Counter(spoken_tokens)
        matched = 0
        for tok in ref_tokens:
            if spoken_counts.get(tok, 0) > 0:
                spoken_counts[tok] -= 1
                matched += 1
        content = matched / len(ref_tokens) * 100.0
    else:
        matched = 0
        content = 0.0

    # ── Speech metrics ──────────────────────────────────────────────────────
    # WPM + speech_dur use Azure word timestamps (or Whisper fallback) —
    # those are aligned with what was actually recognised as speech.
    if azure_words:
        first_start_ms = azure_words[0]["offset_ms"]
        last_end_ms = azure_words[-1]["offset_ms"] + azure_words[-1]["duration_ms"]
        speech_dur = max(0.001, (last_end_ms - first_start_ms) / 1000.0)
        wpm = len(azure_words) * 60.0 / speech_dur
    elif whisper_words:
        speech_dur = max(0.001, whisper_words[-1]["end"] - whisper_words[0]["start"])
        wpm = len(whisper_words) * 60.0 / speech_dur
    else:
        speech_dur = 0.0
        wpm = 0.0

    # Pause detection: pydub continuous-silence ≥ 500 ms below -30 dBFS,
    # within-speech only (leading + trailing excluded with 200 ms tolerance).
    # Each pause carries (start_ms, end_ms) so we can attach word context.
    pause_intervals = []      # list of (start_ms, end_ms) within-speech only
    audio_dur = 0.0
    try:
        from pydub import AudioSegment
        from pydub.silence import detect_silence
        import io as _io
        _seg = AudioSegment.from_file(_io.BytesIO(audio_bytes))
        _total_ms = len(_seg)
        audio_dur = _seg.duration_seconds
        _sils = detect_silence(
            _seg,
            min_silence_len=_RA_PAUSE_MIN_MS,
            silence_thresh=_SILENCE_THRESH_DBFS,
        )
        for s, e in _sils:
            if s <= _RA_PAUSE_LEADING_TOL_MS:
                continue                        # leading dead air
            if (_total_ms - e) <= _RA_PAUSE_TRAILING_TOL_MS:
                continue                        # trailing dead air
            pause_intervals.append((s, e))
    except Exception as e:
        log.warning("[RA_V3] pydub pause detection failed (continuing with 0): %s", e)

    gaps_ms = [int(e - s) for s, e in pause_intervals]
    gap_pauses = len(gaps_ms)

    hesitation_count = sum(
        1 for w in whisper_words
        if _HESITATION_RE.match(w["text"].strip().rstrip(".,!?"))
    )

    total_pauses = gap_pauses + hesitation_count
    sentence_count = _count_sentences(reference_text)

    pause_breakdown, pause_breakdown_overflow = _build_ra_pause_breakdown(
        pause_intervals=pause_intervals,
        whisper_words=whisper_words,
        azure_words=azure_words,
    )

    # ── WPM hard gates (kept) ──────────────────────────────────────────────
    wpm_gate_triggered = wpm < 80.0 or wpm > 250.0

    # ── WPM band score (kept; RA curve) ─────────────────────────────────────
    if wpm_gate_triggered or wpm <= 0:
        wpm_band_score = 0.0
    elif wpm < 130.0:
        wpm_band_score = 100.0 - 2.0 * (130.0 - wpm)
    elif wpm <= 200.0:
        wpm_band_score = 100.0
    else:
        wpm_band_score = max(0.0, 100.0 - 2.0 * (wpm - 200.0))

    # ── Pause penalty score (NEW) ──────────────────────────────────────────
    s_clamped = min(max(sentence_count, 1), 10)  # safe denominator
    if total_pauses < s_clamped:
        pause_penalty_score = 100.0
    elif total_pauses <= 10:
        pause_penalty_score = 100.0 * (10 - total_pauses) / (11 - s_clamped)
    else:
        pause_penalty_score = 0.0
    pause_penalty_score = max(0.0, min(100.0, pause_penalty_score))

    # ── Combine — worst signal wins ─────────────────────────────────────────
    if wpm_gate_triggered:
        fluency = 0.0
    else:
        fluency = min(wpm_band_score, pause_penalty_score)

    # ── Cross-penalty (existing) ───────────────────────────────────────────
    content_pre, fluency_pre, pron_pre = content, fluency, pronunciation
    mC = _cross_multiplier(min(fluency, pronunciation))
    mF = _cross_multiplier(min(content, pronunciation))
    mP = _cross_multiplier(min(content, fluency))
    if mC < 1.0 or mP < 1.0:
        content = max(0.0, content * mC)
        fluency = max(0.0, fluency * mF)
        pronunciation = max(0.0, pronunciation * mP)
        log.info(
            "[CROSS_PENALTY] q=%s type=read_aloud user=%s "
            "mC=%.3f mF=%.3f mP=%.3f before c/f/p=%.1f/%.1f/%.1f → after c/f/p=%.1f/%.1f/%.1f",
            question_id, user_id, mC, mF, mP,
            content_pre, fluency_pre, pron_pre, content, fluency, pronunciation,
        )

    log.info(
        "[RA_V2] q=%s user=%s words_ref=%d words_whisper=%d matched=%d "
        "wpm=%.1f speech_dur=%.2fs gap_pauses=%d hesitations=%d total_pauses=%d "
        "sentences=%d wpm_band=%.1f pause_score=%.1f gate=%s "
        "→ c/f/p=%.1f/%.1f/%.1f",
        question_id, user_id, len(ref_tokens), len(spoken_tokens), matched,
        wpm, speech_dur, gap_pauses, hesitation_count, total_pauses,
        sentence_count, wpm_band_score, pause_penalty_score, wpm_gate_triggered,
        content, fluency, pronunciation,
    )

    fluency_metrics = {
        "wpm":                       round(wpm, 1),
        "speech_duration_sec":       round(speech_dur, 2),
        "audio_duration_sec":        round(audio_dur, 2),
        "gap_pauses":                gap_pauses,
        "hesitation_count":          hesitation_count,
        "total_pauses":              total_pauses,
        "sentence_count":            sentence_count,
        "pause_lengths_ms":          gaps_ms,
        "pause_threshold_ms":        _RA_PAUSE_MIN_MS,
        "pause_method":              "pydub_silence",
        "pause_breakdown":           pause_breakdown,
        "pause_breakdown_overflow":  pause_breakdown_overflow,
        "wpm_band_score":            round(wpm_band_score, 1),
        "pause_penalty_score":       round(pause_penalty_score, 1),
        "wpm_gate_triggered":        wpm_gate_triggered,
        "matched_tokens":            matched,
        "ref_token_count":           len(ref_tokens),
        "spoken_token_count":        len(spoken_tokens),
        "cross_multipliers":         {"mC": round(mC, 3), "mF": round(mF, 3), "mP": round(mP, 3)},
    }

    return {
        "content":         round(content, 1),
        "fluency":         fluency,
        "pronunciation":   pronunciation,
        "transcript":      transcript,
        "word_scores":     word_scores,
        "fluency_metrics": fluency_metrics,
    }


def _transcribe_azure_with_whisper_parallel(audio_bytes: bytes) -> dict:
    """
    Run Azure transcribe_and_score_free + Whisper transcription concurrently.
    Returns Azure's full result dict, with 'transcript' replaced by Whisper's
    output when available. Adds audit fields:
      - 'azure_transcript':   raw Azure transcript
      - 'whisper_transcript': raw Whisper transcript ('' if failed)
      - 'transcript_source':  'whisper' | 'azure_fallback' | 'none'

    Whisper runs in a daemon thread that joins after Azure's longer call
    returns. Net latency is max(Azure, Whisper) instead of sum.

    Used by the LLM-content-scored branches (DI, RL, RTS, ptea_RTS, SGD)
    where Azure's en-US ASR mishears domain vocabulary. Whisper's broader
    language model handles technical PTE terms (Radiata, chipper, fallout)
    and accented English better.
    """
    from services.azure_speech_service import transcribe_and_score_free

    holder = {}

    def _whisper_worker():
        try:
            from services.whisper_service import transcribe_with_whisper
            holder['whisper'] = transcribe_with_whisper(audio_bytes)
        except Exception as e:
            log.error("[WHISPER] worker thread failed: %s", e)
            holder['whisper'] = ""

    t = threading.Thread(target=_whisper_worker, daemon=True)
    t.start()

    azure_result = transcribe_and_score_free(audio_bytes)
    azure_transcript = azure_result.get('transcript', '') if isinstance(azure_result, dict) else ''

    # Cap Whisper wait at 15s in case the API hangs.
    t.join(timeout=15.0)
    whisper_transcript = holder.get('whisper', '')

    if whisper_transcript:
        chosen = whisper_transcript
        source = 'whisper'
    elif azure_transcript:
        chosen = azure_transcript
        source = 'azure_fallback'
    else:
        chosen = ''
        source = 'none'

    log.info(
        "[WHISPER] azure_len=%d whisper_len=%d source=%s",
        len(azure_transcript), len(whisper_transcript), source,
    )

    return {
        **(azure_result if isinstance(azure_result, dict) else {}),
        'transcript': chosen,
        'azure_transcript': azure_transcript,
        'whisper_transcript': whisper_transcript,
        'transcript_source': source,
    }


def _get_stimulus_key_points(question_type: str, audio_url: str) -> list:
    """Transcribe stimulus audio + GPT-extract key points for RL/RTS/SGD."""
    try:
        from services.azure_speech_service import transcribe_audio_full
        from services.llm_content_scoring_service import extract_key_points
        presigned = generate_presigned_url(audio_url)
        resp = _requests.get(presigned, timeout=30)
        resp.raise_for_status()
        transcript = transcribe_audio_full(resp.content)
        if transcript:
            return extract_key_points(transcript, question_type)
    except Exception as e:
        log.error(f"[SCORER] Stimulus key-point extraction failed ({question_type}): {e}")
    return []


def _run_scoring(
    user_id: int,
    question_id: int,
    question_type: str,
    audio_url: str,
    reference_text: str = "",
    key_points: list = None,
    expected_answers: list = None,
    stimulus_audio_url: str = "",
):
    from services.azure_speech_service import score_read_aloud, transcribe_and_score_free

    if key_points is None:
        key_points = []
    if expected_answers is None:
        expected_answers = []

    try:
        presigned = generate_presigned_url(audio_url)
        audio_resp = _requests.get(presigned, timeout=30)
        audio_resp.raise_for_status()
        audio_bytes = audio_resp.content

        content = 0.0
        fluency = 0.0
        pronunciation = 0.0
        transcript = ""
        word_scores = []
        content_llm_scored = False
        extra = {}

        if question_type == "read_aloud" and reference_text:
            # New RA pipeline: Whisper-driven content (positional) + pause-
            # based fluency. Skips _apply_speaking_fluency_formula entirely
            # — RA persists fluency_metrics and applies cross-penalty inside
            # _score_read_aloud_v2.
            raw = _score_read_aloud_v2(
                user_id=user_id,
                question_id=question_id,
                audio_bytes=audio_bytes,
                reference_text=reference_text,
            )
            content        = raw["content"]
            fluency        = raw["fluency"]
            pronunciation  = raw["pronunciation"]
            transcript     = raw.get("transcript", "")
            word_scores    = raw.get("word_scores", [])
            extra          = {"_ra_v2_metrics": raw.get("fluency_metrics", {})}

        elif question_type == "repeat_sentence" and reference_text:
            raw = score_read_aloud(audio_bytes, reference_text)
            content       = raw["content"]
            fluency       = raw["fluency"]
            pronunciation = raw["pronunciation"]
            transcript    = raw.get("transcript", "")
            word_scores   = raw.get("word_scores", [])

        elif question_type == "answer_short_question":
            raw = transcribe_and_score_free(audio_bytes)
            transcript    = raw.get("transcript", "")
            fluency       = raw.get("fluency", 0)
            pronunciation = raw.get("pronunciation", 0)
            word_scores   = raw.get("word_scores", [])
            is_correct = False
            if expected_answers:
                t_lower = transcript.lower().strip()
                # Match each accepted variant as a whole phrase with word
                # boundaries. Any variant match → correct. The DB-provided
                # acceptedVariants list already covers article/phrasing
                # alternatives ("diaper", "a diaper", "the diaper"), so we
                # don't need to split the variant into per-word tokens —
                # which would re-introduce false positives like "a sheep"
                # matching "a banana" on the article alone.
                for ans in expected_answers:
                    ans_lower = ans.lower().strip()
                    if not ans_lower:
                        continue
                    if re.search(rf'\b{re.escape(ans_lower)}\b', t_lower):
                        is_correct = True
                        break
            content = 100.0 if is_correct else 0.0
            extra = {"is_correct": is_correct}

        elif question_type == "describe_image":
            # Whisper transcribes in parallel with Azure for better content
            # scoring on PTE-specific vocab + accented English. Azure still
            # provides word_scores, fluency, pronunciation as before.
            raw = _transcribe_azure_with_whisper_parallel(audio_bytes)
            transcript    = raw.get("transcript", "")
            fluency       = raw.get("fluency", 0)
            pronunciation = raw.get("pronunciation", 0)
            word_scores   = raw.get("word_scores", [])
            if key_points and transcript:
                from services.llm_content_scoring_service import score_content_with_llm
                content = float(score_content_with_llm(transcript, key_points, question_type))
                content_llm_scored = True
            else:
                content = min(len(transcript.split()) / 40, 1.0) * 50.0 if transcript else 0.0

        else:
            # RL, RTS, ptea_RTS, SGD — Whisper-parallel transcription too.
            raw = _transcribe_azure_with_whisper_parallel(audio_bytes)
            transcript    = raw.get("transcript", "")
            fluency       = raw.get("fluency", 0)
            pronunciation = raw.get("pronunciation", 0)
            word_scores   = raw.get("word_scores", [])
            if not key_points and stimulus_audio_url:
                key_points = _get_stimulus_key_points(question_type, stimulus_audio_url)
            if key_points and transcript:
                from services.llm_content_scoring_service import score_content_with_llm
                content = float(score_content_with_llm(transcript, key_points, question_type))
                content_llm_scored = True
            else:
                _targets = {
                    "retell_lecture": 60,
                    "respond_to_situation": 30,
                    "summarize_group_discussion": 50,
                }
                target = _targets.get(question_type, 40)
                content = min(len(transcript.split()) / target, 1.0) * 50.0 if transcript else 0.0

        # Speaking fluency formula replaces Azure's FluencyScore for the
        # 7 types in _FLUENCY_FORMULA_TYPES across practice + sectional + mock.
        # Content (CompletenessScore) and pronunciation (AccuracyScore) pass
        # through Azure-as-is. ASQ is excluded. RA uses its own v2 path
        # (_score_read_aloud_v2) which already computes fluency + cross-
        # penalty + metrics, so it short-circuits here.
        if "_ra_v2_metrics" in extra:
            fluency_metrics = extra.pop("_ra_v2_metrics")
        else:
            content, fluency, pronunciation, fluency_metrics = _apply_speaking_fluency_formula(
                user_id=user_id,
                question_id=question_id,
                question_type=question_type,
                audio_bytes=audio_bytes,
                word_scores=word_scores,
                content=content,
                fluency=fluency,
                pronunciation=pronunciation,
                reference_text=reference_text,
                transcript=transcript,
            )

        # Rubric-weighted PTE score (uniform for all types)
        computed = _compute_question_score(question_type, {
            "content": content,
            "fluency": fluency,
            "pronunciation": pronunciation,
            "content_llm_scored": content_llm_scored,
            **extra,
        })
        pte = _pte_score(computed["pct"])

        store_score(user_id, question_id, {
            "scoring":       "complete",
            "content":       round(content, 1),
            "fluency":       fluency,
            "pronunciation": pronunciation,
            "total":         max(config.PTE_FLOOR, pte),
            "transcript":    transcript,
            "word_scores":   word_scores,
            "fluency_metrics": fluency_metrics,
            **extra,
        })
        update_speaking_score_in_db(
            user_id=user_id,
            question_id=question_id,
            content=round(content, 1),
            pronunciation=pronunciation,
            fluency=fluency,
            total=pte,
            transcript=transcript,
            word_scores=word_scores,
            fluency_metrics=fluency_metrics,
        )
        log.info(f"[SCORER] q={question_id} type={question_type} content={content:.1f} " f"fluency={fluency} pronunciation={pronunciation} pte={pte}")
        if question_type == "answer_short_question":
            log.info(f"[ASQ] q={question_id} " f"user_said={transcript!r} " f"expected={expected_answers!r} " f"is_correct={extra.get('is_correct')} " f"pte={pte}")

    except Exception as e:
        store_score(user_id, question_id, {"scoring": "error", "error": str(e)})
        if "no speech recognised" in str(e).lower():
            logger.warning(
                "[SCORER] user=%s question=%s type=%s no speech recognised",
                user_id, question_id, question_type,
            )
        else:
            logger.error(
                "[SCORER ERROR] user=%s question=%s type=%s exception=%s: %s",
                user_id, question_id, question_type, type(e).__name__, e,
            )
        # Always mark AttemptAnswer complete so background aggregation is never blocked
        update_speaking_score_in_db(
            user_id=user_id,
            question_id=question_id,
            content=0.0,
            pronunciation=0.0,
            fluency=0.0,
            total=0,
            transcript="",
            word_scores=[],
        )


def kick_off_scoring(
    user_id: int,
    question_id: int,
    question_type: str,
    audio_url: str,
    reference_text: str = "",
    key_points: list = None,
    expected_answers: list = None,
    stimulus_audio_url: str = "",
):
    t = threading.Thread(
        target=_run_scoring,
        args=(user_id, question_id, question_type, audio_url,
              reference_text, key_points, expected_answers, stimulus_audio_url),
        daemon=True,
    )
    t.start()
