# EnglishFirm Web FastAPI — Claude Workflow Rules

## Git Workflow (MANDATORY — follow every time)

Before making ANY code change:
1. `git pull origin main` — always pull latest first
2. Make the change
3. `git add <specific files>` + `git commit -m "..."`
4. `git push origin main`

Never edit files without pulling first. Never leave uncommitted changes. No exceptions.

## Secret Files — NEVER COMMIT (MANDATORY)

NEVER stage or commit these files under any circumstances:
- `.env` (any variant: `.env.local`, `.env.production`, etc.)
- Any file containing API keys, passwords, or credentials
- `google-services.json`, `GoogleService-Info.plist`

Always use `git add <specific files>` — never `git add .` or `git add -A`.
Before every commit, verify with `git diff --cached --name-only` that no secret files are staged.

## Repos
- **This repo (web backend):** `englishfirm-web-fastapi` — `/Users/kaspintonus/Desktop/App/git/englishfirm-web-fastapi`
- **Web frontend:** `englishfirm-web-flutter` — `/Users/kaspintonus/Desktop/App/git/englishfirm-web-flutter`
- **Mobile backend reference:** `englishfirm-app-fastapi` — `/Users/kaspintonus/Desktop/App/git/englishfirm-app-fastapi`
- **Mobile frontend reference:** `englishfirm-app-flutter` — `/Users/kaspintonus/Desktop/App/git/englishfirm-app-flutter`

All backend code for the web app goes here — never commit web backend changes to `englishfirm-app-fastapi`.

## Reference Backend

When building APIs, refer to `englishfirm-app-fastapi` for:
- Existing API contracts (endpoint paths, request/response shapes)
- DB models and schema (`db/models.py`)
- Scoring logic (`services/question_service.py`, `services/speaking_scoring_service.py`)
- Azure speech integration (`services/azure_speech_service.py`)
- S3 integration (`services/s3_service.py`)
- Session management patterns (`ACTIVE_SESSIONS`, `_SCORE_STORE`)

## Key Infrastructure
- DB: PostgreSQL (same RDS instance as mobile backend)
- Audio storage: S3 bucket `apeuni-user-recordings`
- Questions: S3 bucket `apeuni-questions-audio`
- Azure Speech scoring: region `australiaeast`, key in `.env`

## AWS Production Infrastructure (Reference)

Production runs on AWS in `ap-southeast-2`. Use `--profile englishfirm` on every AWS CLI call.

### Coordinates
| What | Value |
|---|---|
| AWS account | `549209747198` |
| Region | `ap-southeast-2` |
| AWS CLI profile | `englishfirm` |
| Backend domain | `https://api.englishfirm.com` |
| Frontend domain | `https://app.englishfirm.com` |
| Backend health | `curl https://api.englishfirm.com/health` |

### Backend (FastAPI on ECS Fargate)
| What | Value |
|---|---|
| ECS cluster | `englishfirm-prod` |
| ECS service | `englishfirm-web-fastapi` |
| Task def family | `englishfirm-web-fastapi` |
| CloudWatch log group | `/ecs/englishfirm-web-fastapi` |
| ECR repo | `549209747198.dkr.ecr.ap-southeast-2.amazonaws.com/englishfirm-web-fastapi` |
| ALB | `englishfirm-web-alb-477597736.ap-southeast-2.elb.amazonaws.com` |
| Target group | `englishfirm-web-tg` |
| Task SG | `sg-052e0ea449131e68c` |
| Secrets Manager | `englishfirm-web-fastapi/prod` |

### Frontend
| What | Value |
|---|---|
| S3 bucket | `englishfirm-web-flutter` |
| CloudFront distro | `E2JXAOH26OS3MK` (`d3msnux4tulesc.cloudfront.net`) |

### Database
| What | Value |
|---|---|
| RDS instance | `database-1` (Postgres 17, `db.t3.micro`) |
| RDS endpoint | `database-1.cdw0ciyucyrd.ap-southeast-2.rds.amazonaws.com:5432` |
| DB name | `postgres` |
| Shared with | mobile backend (`englishfirm-app-fastapi`) — see schema rules in `scripts/deploy/RUNBOOK.md` |

> The DB password lives in `.env` (never commit). Do NOT paste it into commit messages or PR bodies.

### S3 buckets
| Bucket | Purpose |
|---|---|
| `apeuni-user-recordings` | User audio uploads (speaking attempts) |
| `apeuni-questions-audio` | Question stimulus audio |
| `englishfirm-web-flutter` | Flutter web build artifacts |

### Common log + status commands

```bash
# Tail backend logs (live)
aws logs tail /ecs/englishfirm-web-fastapi --follow \
  --profile englishfirm --region ap-southeast-2

# Tail with filter (e.g. scoring markers)
aws logs tail /ecs/englishfirm-web-fastapi --follow \
  --filter-pattern "FLUENCY_FORMULA" \
  --profile englishfirm --region ap-southeast-2

# Last 30 min of logs (no follow)
aws logs tail /ecs/englishfirm-web-fastapi --since 30m \
  --profile englishfirm --region ap-southeast-2

# Service status (rollout, running tasks, current task def)
aws ecs describe-services --cluster englishfirm-prod \
  --services englishfirm-web-fastapi \
  --profile englishfirm --region ap-southeast-2 \
  --query 'services[0].deployments[?status==`PRIMARY`] | [0].{rollout:rolloutState,running:runningCount,desired:desiredCount,td:taskDefinition,failed:failedTasks}'

# List task def revisions (latest first)
aws ecs list-task-definitions --family-prefix englishfirm-web-fastapi \
  --sort DESC --max-items 10 \
  --profile englishfirm --region ap-southeast-2

# Health check
curl -fsS https://api.englishfirm.com/health
```

### Deploy / rollback
- Deploy / rollback scripts live in `scripts/deploy/` — see `RUNBOOK.md` for the full procedure.
- Never deploy or rollback without explicit user instruction.

### Safety
- AWS CLI calls that READ (`describe-*`, `list-*`, `logs tail`, `get-*`) are safe to run for diagnostics.
- AWS CLI calls that WRITE (`update-service`, `register-task-definition`, `delete-*`, `terminate-*`, `put-*`, S3 `cp`/`rm`/`sync` to a prod bucket) are NOT safe — confirm with the user before running.
- Never check AWS keys (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`) into git or paste them into chat history. They live in `~/.aws/credentials` under profile `englishfirm`.

## Cross-Platform Guardrail (MANDATORY)

All API endpoints must remain compatible with the iOS app (`englishfirm-app-flutter`).
- Never change a response field name, status code, or response structure without verifying both clients handle it
- Never add a required request field that the iOS client doesn't send
- Backend logic must be platform-agnostic — no assumptions about the calling platform

## Scoring Guardrail (MANDATORY)

All scoring logic lives in `services/scoring/`. Never write inline scoring logic in routers or other services.

### Scoring module structure
```
services/scoring/
  base.py        — ScoringResult dataclass, ScoringStrategy ABC, to_pte_score()
  rule_scorer.py — FIBScorer, MCQScorer, ReorderScorer, WFDScorer, HIWScorer (sync)
  ai_scorer.py   — AIScorer for SWT, WE, SST (sync Claude/GPT call)
  azure_scorer.py— AzureSpeakingScorer for all speaking types (async, wraps thread)
  aggregator.py  — SectionalAggregator: waits for all question scores, computes final PTE
  registry.py    — get_scorer(question_type) → ScoringStrategy instance
```

### PTE Score Formula (MANDATORY — no exceptions)
```python
def to_pte_score(weighted_pct: float) -> int:
    return max(10, min(90, round(10 + weighted_pct * 80)))
```
- Floor: 10, Ceiling: 90, Scale: 80
- Every score display must use this formula

### Rules
- `get_scorer(question_type)` → call the returned scorer for ALL question scoring
- Practice mode: `scorer.score(question_id, session_id, answer)` → returns `ScoringResult` immediately (sync) or pending (async for speaking)
- Sectional mode: same per-question scorer + `SectionalAggregator.aggregate()` at finish
- `ScoringResult.is_async == True` → speaking types; client must poll
- `ScoringResult.is_async == False` → all other types; return score immediately in submit response (Option 4)
- Never create a new scorer class for a type already in the registry — extend or reuse
- Never duplicate the `_RUBRIC` or `_SPEAKING_WEIGHTS` tables — they live in `azure_scorer.py` only
- Never write `max(10, min(90, ...))` inline — always call `to_pte_score()`
- Scoring must work identically for practice mode and sectional mode at the question level

### Reuse map
| Question Type | Scorer | Sync? |
|---|---|---|
| read_aloud, repeat_sentence, describe_image, retell_lecture, summarize_group_discussion, respond_to_situation, answer_short_question | `AzureSpeakingScorer` | No — poll |
| reading_mcs, reading_mcm, listening_mcs, listening_mcm, listening_hcs, listening_smw | `MCQScorer` | Yes |
| reading_fib, listening_fib | `FIBScorer` | Yes |
| reading_fib_drop_down | `FIBScorer` | Yes |
| reorder_paragraphs | `ReorderScorer` | Yes |
| listening_wfd | `WFDScorer` | Yes |
| listening_hiw | `HIWScorer` | Yes |
| summarize_written_text, write_essay | `AIScorer` | Yes |
| listening_sst | `AIScorer` | Yes |

## Router Structure Guardrail (MANDATORY)

All per-type routers live in `routers/`. One file per question type. Mirror this structure exactly.

```
routers/
  speaking/
    read_aloud.py                 → /api/v1/questions/speaking/read-aloud
    repeat_sentence.py            → /api/v1/questions/speaking/repeat-sentence
    describe_image.py             → /api/v1/questions/speaking/describe-image
    retell_lecture.py             → /api/v1/questions/speaking/retell-lecture
    summarize_group_discussion.py → /api/v1/questions/speaking/summarize-group-discussion
    respond_to_situation.py       → /api/v1/questions/speaking/respond-to-situation
    answer_short_question.py      → /api/v1/questions/speaking/answer-short-question
  writing/
    summarize_written_text.py     → /api/v1/questions/writing/summarize-written-text
    write_essay.py                → /api/v1/questions/writing/write-essay
  reading/
    mcs.py                        → /api/v1/questions/reading/mcs
    mcm.py                        → /api/v1/questions/reading/mcm
    fill_in_blanks.py             → /api/v1/questions/reading/fib
    fib_drag_drop.py              → /api/v1/questions/reading/fib-drag-drop
    reorder_paragraphs.py         → /api/v1/questions/reading/reorder-paragraphs
  listening/
    mcs.py                        → /api/v1/questions/listening/mcs
    mcm.py                        → /api/v1/questions/listening/mcm
    fib.py                        → /api/v1/questions/listening/fib
    hcs.py                        → /api/v1/questions/listening/hcs
    smw.py                        → /api/v1/questions/listening/smw
    hiw.py                        → /api/v1/questions/listening/hiw
    wfd.py                        → /api/v1/questions/listening/wfd
    sst.py                        → /api/v1/questions/listening/sst
  sectional/
    speaking.py                   → /api/v1/questions/sectional/speaking
    writing.py                    → /api/v1/questions/sectional/writing
    reading.py                    → /api/v1/questions/sectional/reading
    listening.py                  → /api/v1/questions/sectional/listening
```

### Standard endpoint pattern per router

| Endpoint | All types | Speaking only |
|---|---|---|
| `POST /start` | ✓ | ✓ |
| `POST /submit` | ✓ | ✓ |
| `GET /score/{question_id}` | — | ✓ poll Azure score |
| `GET /audio-url` | — | ✓ presigned playback URL |
| `GET /upload-url` | — | ✓ presigned S3 upload URL |

### Rules
- Every `POST /submit` MUST call `get_scorer(question_type).score(...)` — never write inline scoring in a router
- To add a new question type: create a new file, add its scorer to `services/scoring/registry.py`, mount in `main.py`
- Never change endpoint paths without verifying the iOS client (`englishfirm-app-flutter`) still works
- Session state lives in `services/session_service.py` — never import `ACTIVE_SESSIONS` or `_SCORE_STORE` directly from any other module
- Sectional routers at `routers/sectional/` own exam-level flows (finish + aggregate) — per-question scoring still delegates to `get_scorer()`

## Config & Code Hygiene Guardrail (MANDATORY)

All non-secret constants live in `core/config.py`. Never write magic strings or numbers anywhere else.

### What lives in `core/config.py`
| Constant | Example |
|---|---|
| S3 bucket names | `S3_RECORDINGS_BUCKET`, `S3_QUESTIONS_BUCKET` |
| AWS region | `AWS_REGION` |
| Azure region | `AZURE_SPEECH_REGION` |
| Presigned URL expiry times | `PRESIGNED_READ_EXPIRY_SECONDS`, `PRESIGNED_UPLOAD_EXPIRY_SECONDS` |
| DB pool settings | `DB_POOL_SIZE`, `DB_MAX_OVERFLOW` |
| JWT algorithm | `JWT_ALGORITHM` |
| Session question limit | `SESSION_QUESTION_LIMIT` |
| PTE score formula constants | `PTE_FLOOR`, `PTE_CEILING`, `PTE_BASE`, `PTE_SCALE` |

### What lives in `.env` (secrets — never in `core/config.py`)
`DATABASE_URL`, `JWT_SECRET_KEY`, `AZURE_SPEECH_KEY`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`
`core/config.py` reads these via `os.getenv()` but never defines their values.

### DB table/column names
Must only appear in `db/models.py`. Never reference a table or column name string outside a model definition.

### Rules
- Before writing any string literal or number in a router or service — check if it belongs in `core/config.py`
- Never import `os.getenv(...)` directly in a router or service — always import from `core/config`
- `_RUBRIC` and `_SPEAKING_WEIGHTS` stay in `services/scoring/azure_scorer.py` — they are domain logic, not config
