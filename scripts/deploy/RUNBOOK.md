# Deployment & Rollback Runbook

**Last verified:** 2026-05-01

## Quick reference

| Environment | URL | Where it runs |
|---|---|---|
| Backend (prod) | https://api.englishfirm.com | ECS Fargate (cluster `englishfirm-prod`) in `ap-southeast-2` |
| Frontend (prod) | https://app.englishfirm.com | CloudFront → S3 `englishfirm-web-flutter` |
| Database | `database-1.cdw0ciyucyrd.ap-southeast-2.rds.amazonaws.com:5432` | RDS Postgres 17 (`db.t3.micro`) |
| WordPress | https://englishfirm.com | SiteGround (untouched) |

## Branch model
- `dev` — active development, runs locally
- `main` — production code, deploys via CI/CD or scripts below

---

## DEPLOY backend (web FastAPI)

```bash
cd ~/Desktop/App/git/englishfirm-web-fastapi
git checkout main && git pull
./scripts/deploy/build-and-push.sh
```
Runs: build linux/amd64 → push ECR (tagged with git SHA) → register new task def → update ECS service → wait stable → verify `/health`.
Expected duration: 4–8 min.

## ROLLBACK backend

```bash
cd ~/Desktop/App/git/englishfirm-web-fastapi
./scripts/deploy/rollback.sh           # interactive
./scripts/deploy/rollback.sh 5         # roll back to revision 5
```
Lists last 10 task def revisions. Pick one. ECS does a rolling replacement back to that image.
Expected duration: 2–5 min.

## DEPLOY frontend (Flutter web)

```bash
cd ~/Desktop/App/git/englishfirm-web-flutter
git checkout main && git pull
export CF_DISTRIBUTION_ID=<id>          # one-time, or set in shell rc
./scripts/deploy/build-and-deploy.sh
```
Runs: `flutter build web --release` → upload to `s3://englishfirm-web-flutter/builds/<sha>/` → update CloudFront origin path → invalidate cache.
Expected duration: 3–6 min build + 5–10 min CloudFront propagation.

## ROLLBACK frontend

```bash
cd ~/Desktop/App/git/englishfirm-web-flutter
export CF_DISTRIBUTION_ID=<id>
./scripts/deploy/rollback.sh           # interactive — lists builds in S3
./scripts/deploy/rollback.sh abc1234   # roll back to specific git SHA
```
Changes CloudFront origin path back to old SHA + invalidates cache.
Expected duration: 5–10 min CloudFront propagation.

---

## DATABASE — never drop or rename without coordination

The web backend SHARES the `postgres` database (31 tables) with the mobile backend. Schema rule:
- ✅ Add columns/tables (backwards compatible)
- ✅ Add indexes
- ❌ NEVER drop or rename columns/tables in the same release
- ❌ NEVER `TRUNCATE` or destructive `DELETE`

Pattern for removing a column:
1. Release N: stop writing to it
2. Release N+1: stop reading from it
3. Release N+2: drop it (separate migration, after all clients deployed)

Backups: RDS automated 7-day backups + manual snapshot `pre-ecs-deployment-2026-05-01`.

---

## INCIDENT — backend is down

1. Check ECS service: `aws ecs describe-services --cluster englishfirm-prod --services englishfirm-web-fastapi --profile englishfirm --region ap-southeast-2`
2. Check task health: target group `englishfirm-web-tg` → AWS console → unhealthy targets explain why
3. Check container logs: `aws logs tail /ecs/englishfirm-web-fastapi --follow --profile englishfirm --region ap-southeast-2`
4. If recent deploy: rollback via script above
5. If DB issue: check RDS status + connection count

## INCIDENT — frontend is down (white screen / 404)

1. Check CloudFront distribution status (Console → CloudFront → distribution)
2. Check S3 has files at current origin path
3. Check Cloudflare DNS still points to CloudFront
4. If recent deploy: rollback via script above
5. Browser dev tools console — look for missing assets or CSP errors

## INCIDENT — DNS / cert expired

ACM auto-renews if validation CNAMEs are still at Cloudflare. Don't delete the validation records.

---

## KEY RESOURCE IDS

```
AWS account:           549209747198
Region (primary):      ap-southeast-2
ECS cluster:           englishfirm-prod
ECS service:           englishfirm-web-fastapi
ECR repo:              englishfirm-web-fastapi
ALB:                   englishfirm-web-alb-477597736.ap-southeast-2.elb.amazonaws.com
Target group:          englishfirm-web-tg
ALB SG:                sg-0a03b5e0bedbdfaf3
Task SG:               sg-052e0ea449131e68c
Secrets Manager:       englishfirm-web-fastapi/prod
ACM cert (Sydney):     arn:aws:acm:ap-southeast-2:549209747198:certificate/5964d28e-21c3-4ff6-a05b-4b32401900cd
ACM cert (us-east-1):  arn:aws:acm:us-east-1:549209747198:certificate/c4c9ebf7-7a1c-4df7-a4f1-407ae8278105
S3 bucket (frontend):  englishfirm-web-flutter
RDS instance:          database-1
DB shared with:        mobile backend (englishfirm-app-fastapi)
RDS snapshot (safety): pre-ecs-deployment-2026-05-01
```

---

## DECOMMISSIONING the old EC2 instances

When ready (after web traffic confirmed stable on ECS):
1. Verify mobile backend is no longer reading from these EC2s
2. Stop EC2 v1 (`i-0a2df7f2a3e087016`) → wait 24h → terminate
3. Stop EC2 v2 (`i-064a7265759a970b2`) → wait 24h → terminate
4. After EC2s gone, tighten RDS SG: remove `0.0.0.0/0` rule, keep only ECS task SG + dev IPs

This is a separate task — don't rush.
