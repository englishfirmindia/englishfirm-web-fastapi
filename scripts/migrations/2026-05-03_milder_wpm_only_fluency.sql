-- Migration: lower mild_slow penalty to 25
-- Date: 2026-05-03 (third update on the same day)
-- Applied to: postgres database on database-1 (ap-southeast-2)
--
-- Pairs with a code change in services/speaking_scorer.py that limits the
-- subtract penalty to fluency only (content + pronunciation untouched).
-- zero_out kill switch unchanged — still wipes all three subscores.

BEGIN;

UPDATE wpm_scoring_rules SET penalty = 25 WHERE id = 3 AND label = 'mild slow';

COMMIT;

-- Bands after this migration:
--   <80           zero_out  penalty=0   → all subscores → 0 (kill switch)
--   [80, 90)      subtract  penalty=35  → 'slow'        (only fluency hit)
--   [90, 170)     subtract  penalty=25  → 'mild slow'   (only fluency hit)
--   [170, 200)    subtract  penalty=0   → 'ideal'
--   >=200         subtract  penalty=20  → 'too fast'    (only fluency hit)
