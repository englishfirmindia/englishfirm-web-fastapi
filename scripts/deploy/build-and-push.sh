#!/usr/bin/env bash
# Build linux/amd64 image, tag with git SHA + latest, push to ECR, update ECS service.
# Usage: ./scripts/deploy/build-and-push.sh
set -euo pipefail

REGION=ap-southeast-2
ACCOUNT=549209747198
ECR_URI="${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/englishfirm-web-fastapi"
CLUSTER=englishfirm-prod
SERVICE=englishfirm-web-fastapi
PROFILE="${AWS_PROFILE:-englishfirm}"

cd "$(git rev-parse --show-toplevel)"

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "ERROR: working tree has uncommitted changes. Commit or stash first." >&2
  exit 1
fi

SHA=$(git rev-parse --short HEAD)
BRANCH=$(git rev-parse --abbrev-ref HEAD)

if [ "$BRANCH" != "main" ]; then
  read -p "Not on main (current: $BRANCH). Continue? [y/N] " -n 1 -r
  echo
  [[ $REPLY =~ ^[Yy]$ ]] || exit 1
fi

echo "→ Building image: ${ECR_URI}:${SHA}"
aws ecr get-login-password --profile "$PROFILE" --region "$REGION" \
  | docker login --username AWS --password-stdin "${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com"

docker buildx build --platform linux/amd64 \
  -t "${ECR_URI}:${SHA}" \
  -t "${ECR_URI}:latest" \
  --push .

echo "→ Updating task definition image to :${SHA}"
TASK_DEF=$(aws ecs describe-task-definition --task-definition "$SERVICE" \
  --profile "$PROFILE" --region "$REGION" --query 'taskDefinition' --output json)

NEW_DEF=$(echo "$TASK_DEF" | python3 -c "
import json, sys
d = json.load(sys.stdin)
for k in ('taskDefinitionArn','revision','status','requiresAttributes','compatibilities','registeredAt','registeredBy'):
    d.pop(k, None)
for c in d['containerDefinitions']:
    if c['name'] == 'web':
        c['image'] = '${ECR_URI}:${SHA}'
print(json.dumps(d))
")

REV=$(echo "$NEW_DEF" | aws ecs register-task-definition --cli-input-json file:///dev/stdin \
  --profile "$PROFILE" --region "$REGION" --query 'taskDefinition.revision' --output text)

echo "→ Registered task def revision: ${SERVICE}:${REV}"

echo "→ Updating ECS service to new revision"
aws ecs update-service --cluster "$CLUSTER" --service "$SERVICE" \
  --task-definition "${SERVICE}:${REV}" \
  --profile "$PROFILE" --region "$REGION" --output json --query 'service.{Status:status,Desired:desiredCount}'

echo "→ Waiting for deployment to stabilize (up to 10 min)..."
aws ecs wait services-stable --cluster "$CLUSTER" --services "$SERVICE" \
  --profile "$PROFILE" --region "$REGION"

echo "→ Verifying /health"
curl -fsS https://api.englishfirm.com/health || { echo "HEALTH CHECK FAILED"; exit 1; }
echo
echo "✓ Deployed ${SERVICE}:${REV} (${SHA})"
