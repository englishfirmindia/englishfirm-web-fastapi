#!/usr/bin/env bash
# Roll back ECS service to a previous task definition revision.
# Usage: ./scripts/deploy/rollback.sh <revision-number>
#   ./scripts/deploy/rollback.sh           # interactive — pick from last 10 revisions
#   ./scripts/deploy/rollback.sh 5         # roll back to englishfirm-web-fastapi:5
set -euo pipefail

REGION=ap-southeast-2
CLUSTER=englishfirm-prod
SERVICE=englishfirm-web-fastapi
PROFILE="${AWS_PROFILE:-englishfirm}"

if [ -z "${1:-}" ]; then
  echo "Recent task definition revisions:"
  aws ecs list-task-definitions --family-prefix "$SERVICE" --sort DESC \
    --profile "$PROFILE" --region "$REGION" --max-items 10 \
    --query 'taskDefinitionArns[*]' --output text | tr '\t' '\n' | awk -F: '{print "  rev " $NF}'
  echo
  read -p "Roll back to revision: " REV
else
  REV=$1
fi

CURRENT=$(aws ecs describe-services --cluster "$CLUSTER" --services "$SERVICE" \
  --profile "$PROFILE" --region "$REGION" \
  --query 'services[0].taskDefinition' --output text | awk -F: '{print $NF}')

echo
echo "Cluster:  $CLUSTER"
echo "Service:  $SERVICE"
echo "Current:  ${SERVICE}:${CURRENT}"
echo "Target:   ${SERVICE}:${REV}"
echo
read -p "Confirm rollback? [y/N] " -n 1 -r
echo
[[ $REPLY =~ ^[Yy]$ ]] || { echo "Aborted."; exit 1; }

echo "→ Rolling back..."
aws ecs update-service --cluster "$CLUSTER" --service "$SERVICE" \
  --task-definition "${SERVICE}:${REV}" \
  --force-new-deployment \
  --profile "$PROFILE" --region "$REGION" --output json \
  --query 'service.{Status:status,Desired:desiredCount}'

echo "→ Waiting for service to stabilize..."
aws ecs wait services-stable --cluster "$CLUSTER" --services "$SERVICE" \
  --profile "$PROFILE" --region "$REGION"

echo "→ Verifying /health"
curl -fsS https://api.englishfirm.com/health || { echo "HEALTH CHECK FAILED — investigate"; exit 1; }
echo
echo "✓ Rolled back to ${SERVICE}:${REV}"
