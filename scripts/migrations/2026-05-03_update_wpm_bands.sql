-- Migration: tighten WPM bands for Practice Read Aloud
-- Date: 2026-05-03 (later same day as the initial create)
-- Applied to: postgres database on database-1 (ap-southeast-2)
--
-- Shifts the 'ideal' band UP (was [110, 170), now [170, 200)) and
-- raises penalties for slower bands. Coaching nudge to push faster reading.

BEGIN;

UPDATE wpm_scoring_rules SET min_wpm=NULL, max_wpm=80,   mode='zero_out', penalty=0,  label='very slow → score voided' WHERE id=1;
UPDATE wpm_scoring_rules SET min_wpm=80,   max_wpm=90,   mode='subtract', penalty=35, label='slow'                     WHERE id=2;
UPDATE wpm_scoring_rules SET min_wpm=90,   max_wpm=170,  mode='subtract', penalty=30, label='mild slow'                WHERE id=3;
UPDATE wpm_scoring_rules SET min_wpm=170,  max_wpm=200,  mode='subtract', penalty=0,  label='ideal'                    WHERE id=4;
UPDATE wpm_scoring_rules SET min_wpm=200,  max_wpm=NULL, mode='subtract', penalty=20, label='too fast'                 WHERE id=5;

COMMIT;

-- Bands after this migration:
--   <80           zero_out  → all subscores → 0 (kill switch)
--   [80, 90)      subtract 35  → 'slow'
--   [90, 170)     subtract 30  → 'mild slow' (note: includes typical native pace 130-150)
--   [170, 200)    subtract 0   → 'ideal'
--   >=200         subtract 20  → 'too fast'
