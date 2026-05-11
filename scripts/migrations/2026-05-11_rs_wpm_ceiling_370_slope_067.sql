-- Migration: widen RS WPM band to c=370 / slope=0.67 (gentler high-WPM tail)
-- Date: 2026-05-11
-- Applies to: postgres database on database-1 (ap-southeast-2)
--
-- Today's curve hard-zeros fluency above 270 WPM and below 80, with a
-- steep slope=2.0 across both ascend (80-130) and descend (220-270).
-- Real-world reads above 270 WPM ("native-fluency tongue-twister speed")
-- are rare but legitimate — Nimisha's last sectional RS had two attempts
-- at 282 and 313 WPM scored PTE 34 each (gate triggered) when the rest
-- of her delivery was clean.
--
-- New curve:
--
--   wpm < 80                hard fail   (unchanged)
--   80 ≤ wpm < 130          ascend: 100 − 0.67·(130 − wpm)
--   130 ≤ wpm ≤ 220         plateau 100 (unchanged)
--   220 < wpm ≤ 370         descend: 100 − 0.67·(wpm − 220)
--   wpm > 370               hard fail   (was 270)
--
--                                            today    new
--   band @ 280 WPM (Nimisha's q=5011)           0     58
--   band @ 313 WPM (Nimisha's q=5734)           0     38
--   band @ 350 WPM (genuinely too fast)         0     13
--   band @ 370+ WPM                             0      0  (gate)
--   band @ 100 WPM (mild slow)                 40     80  (also softens)
--
-- Side-effect: slope applies to both ascend and descend. Slow speakers
-- (80-130 WPM) get more credit too. If we later decide that's wrong, the
-- fix is to add a separate descend-slope column — out of scope here.
--
-- Pure data change. _score_speaking_v2 reads both columns at runtime
-- (5-min config cache). No code redeploy needed. Existing
-- attempt_answers rows keep their saved scores; only new submissions get
-- the new curve.
--
-- Idempotent — re-running rewrites the same values.

BEGIN;

UPDATE pte_speaking_scoring_config
SET wpm_ceiling       = 370,
    wpm_slope_per_wpm = 0.67
WHERE task_type = 'repeat_sentence';

COMMIT;

-- Verify:
--   SELECT task_type, wpm_floor, wpm_plateau_low, wpm_plateau_high,
--          wpm_ceiling, wpm_slope_per_wpm
--   FROM pte_speaking_scoring_config
--   WHERE task_type = 'repeat_sentence';
--
-- Expected: 80 / 130 / 220 / 370 / 0.67
--
-- Rollback:
--   UPDATE pte_speaking_scoring_config
--   SET wpm_ceiling = 270, wpm_slope_per_wpm = 2.0
--   WHERE task_type = 'repeat_sentence';
