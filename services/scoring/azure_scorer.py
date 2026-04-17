"""
Azure Speaking scorer — wraps Azure speech scoring for all speaking task types.

Ported from englishfirm-app-fastapi/services/speaking_scoring_service.py.

Module-level constants RUBRIC and SPEAKING_WEIGHTS are exported so
SectionalAggregator can import them directly.
"""

from .base import ScoringResult, ScoringStrategy, to_pte_score

# ── Rubric (mirrors pte_scoring_speaking_rubric) ──────────────────────────────
# total_points = max raw points for this task
# content_pct / fluency_pct / pronunciation_pct must sum to 1.0
RUBRIC = {
    'read_aloud':                 {'total_points': 15, 'content_pct': 0.333,  'fluency_pct': 0.333,  'pronunciation_pct': 0.333},
    'repeat_sentence':            {'total_points': 13, 'content_pct': 0.231,  'fluency_pct': 0.385,  'pronunciation_pct': 0.385},
    'describe_image':             {'total_points': 16, 'content_pct': 0.375,  'fluency_pct': 0.3125, 'pronunciation_pct': 0.3125},
    'retell_lecture':             {'total_points': 16, 'content_pct': 0.375,  'fluency_pct': 0.3125, 'pronunciation_pct': 0.3125},
    'respond_to_situation':       {'total_points': 16, 'content_pct': 0.375,  'fluency_pct': 0.3125, 'pronunciation_pct': 0.3125},
    'ptea_respond_situation':     {'total_points': 16, 'content_pct': 0.375,  'fluency_pct': 0.3125, 'pronunciation_pct': 0.3125},
    'summarize_group_discussion': {'total_points': 16, 'content_pct': 0.375,  'fluency_pct': 0.3125, 'pronunciation_pct': 0.3125},
    'answer_short_question':      {'total_points':  1, 'content_pct': 1.0,    'fluency_pct': 0.0,    'pronunciation_pct': 0.0},
}

# ── Section weights (mirrors pte_question_weightage.speaking_percent) ─────────
# ASQ omitted — speaking_percent is NULL after the DB update
SPEAKING_WEIGHTS = {
    'read_aloud':                 9,
    'repeat_sentence':            16,
    'describe_image':             31,
    'retell_lecture':             13,
    'respond_to_situation':       13,
    'ptea_respond_situation':     13,
    'summarize_group_discussion': 19,
}

# Tasks where content_zero rule applies (RA/RS silence → everything 0)
_CONTENT_ZERO_TASKS = {'read_aloud', 'repeat_sentence'}

# Tasks where content_zero applies only when LLM explicitly scored 0
_CONTENT_ZERO_LLM_TASKS = {
    'describe_image', 'retell_lecture', 'respond_to_situation',
    'summarize_group_discussion',
}


def _compute_question_score(question_type: str, raw_score: dict) -> dict:
    """
    Convert raw score dict → rubric-based points.
    All input scores are on 0-100 scale (raw Azure values or LLM 0-100).
    Returns {'content', 'fluency', 'pronunciation', 'total', 'pct'}.

    Ported exactly from speaking_scoring_service._compute_question_score.
    """
    rubric = RUBRIC.get(question_type)
    if not rubric:
        return {'content': 0.0, 'fluency': 0.0, 'pronunciation': 0.0, 'total': 0.0, 'pct': 0.0}

    max_pts = rubric['total_points']

    # Failed / timed-out scoring
    status = raw_score.get('scoring', '')
    if status in ('error', 'timeout', 'not_found'):
        return {'content': 0.0, 'fluency': 0.0, 'pronunciation': 0.0, 'total': 0.0, 'pct': 0.0}

    # ASQ: binary correct / incorrect
    if question_type == 'answer_short_question':
        correct = raw_score.get('is_correct', False)
        total = 1.0 if correct else 0.0
        return {'content': total, 'fluency': 0.0, 'pronunciation': 0.0, 'total': total, 'pct': total}

    # All scores 0-100
    content_raw       = float(raw_score.get('content', 0) or 0)
    fluency_raw       = float(raw_score.get('fluency', 0) or 0)
    pronunciation_raw = float(raw_score.get('pronunciation', 0) or 0)

    # content_zero rule: RA/RS silence = everything 0
    if content_raw == 0 and question_type in _CONTENT_ZERO_TASKS:
        return {'content': 0.0, 'fluency': 0.0, 'pronunciation': 0.0, 'total': 0.0, 'pct': 0.0}

    # content_zero rule for LLM-scored tasks: only apply when LLM explicitly scored 0
    if (content_raw == 0 and question_type in _CONTENT_ZERO_LLM_TASKS
            and raw_score.get('content_llm_scored')):
        return {'content': 0.0, 'fluency': 0.0, 'pronunciation': 0.0, 'total': 0.0, 'pct': 0.0}

    # Convert 0-100 → rubric points
    content_pts       = (content_raw       / 100.0) * max_pts * rubric['content_pct']
    fluency_pts       = (fluency_raw       / 100.0) * max_pts * rubric['fluency_pct']
    pronunciation_pts = (pronunciation_raw / 100.0) * max_pts * rubric['pronunciation_pct']
    total_pts = content_pts + fluency_pts + pronunciation_pts

    pct = (total_pts / max_pts) if max_pts > 0 else 0.0

    return {
        'content':       round(content_pts, 2),
        'fluency':       round(fluency_pts, 2),
        'pronunciation': round(pronunciation_pts, 2),
        'total':         round(total_pts, 2),
        'pct':           round(pct, 4),
    }


class AzureSpeakingScorer(ScoringStrategy):
    """
    Async scorer for all speaking task types.

    score() kicks off a background Azure thread (via kick_off_fn) and returns
    immediately with is_async=True. The caller polls a separate endpoint for results.

    answer dict:
      audio_url: str
      kick_off_fn: callable(task_type: str, question_id: int, audio_url: str) -> None
    """

    is_async = True  # class-level override (property not used for async scorers)

    def __init__(self, task_type: str):
        self.task_type = task_type

    @property
    def is_async(self) -> bool:
        return True

    def score(self, question_id: int, session_id: str, answer: dict) -> ScoringResult:
        """
        Kicks off background Azure scoring and returns immediately with is_async=True.
        pte_score=0 / raw_score=0.0 are placeholders — client must poll for real score.
        """
        kick_off = answer.get('kick_off_fn')
        if kick_off:
            kick_off(self.task_type, question_id, answer.get('audio_url', ''))
        return ScoringResult(
            pte_score=0,
            raw_score=0.0,
            is_async=True,
            breakdown={'status': 'pending', 'task_type': self.task_type},
        )

    def compute_question_score(self, raw_score: dict) -> dict:
        """
        Port of _compute_question_score from speaking_scoring_service.py.
        Converts raw Azure/LLM scores to rubric-based points for this scorer's task_type.
        """
        return _compute_question_score(self.task_type, raw_score)
