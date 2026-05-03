-- Migration: harsher WPM penalties + lower ideal floor
-- Date: 2026-05-04
-- Applied to: postgres database on database-1 (ap-southeast-2)
--
-- Doubles slow + too_fast penalties (35→50, 20→50). Increases mild_slow
-- penalty 25→40. Lowers ideal floor 170→164 to let comfortable native
-- pace (~165 WPM) escape penalty.
--
-- No code change — rule logic in services/speaking_scorer.py is
-- data-driven and only fluency is penalised (since 2026-05-03).

BEGIN;

UPDATE wpm_scoring_rules SET min_wpm=NULL, max_wpm=80,   mode='zero_out', penalty=0,  label='very slow → score voided' WHERE id=1;
UPDATE wpm_scoring_rules SET min_wpm=80,   max_wpm=90,   mode='subtract', penalty=50, label='slow'                     WHERE id=2;
UPDATE wpm_scoring_rules SET min_wpm=90,   max_wpm=164,  mode='subtract', penalty=40, label='mild slow'                WHERE id=3;
UPDATE wpm_scoring_rules SET min_wpm=164,  max_wpm=200,  mode='subtract', penalty=0,  label='ideal'                    WHERE id=4;
UPDATE wpm_scoring_rules SET min_wpm=200,  max_wpm=NULL, mode='subtract', penalty=50, label='too fast'                 WHERE id=5;

COMMIT;

-- Bands after this migration:
--   <80           zero_out  penalty=0   → kill switch (all 3 → 0)
--   [80, 90)      subtract  penalty=50  → slow         (only fluency hit)
--   [90, 164)     subtract  penalty=40  → mild slow    (only fluency hit)
--   [164, 200)    subtract  penalty=0   → ideal
--   >=200         subtract  penalty=50  → too fast     (only fluency hit)
