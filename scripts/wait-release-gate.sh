#!/usr/bin/env bash
# Refuse every registry write until the complete release-tier workflow has
# succeeded for these exact source bytes. Package workflows start alongside the
# tag gate, so they wait here instead of racing ahead of acceptance.
set -euo pipefail

: "${GITHUB_REPOSITORY:?GITHUB_REPOSITORY is required}"
: "${GITHUB_SHA:?GITHUB_SHA is required}"
: "${GH_TOKEN:?GH_TOKEN is required}"
workflow="${1:-release.yml}"

for attempt in $(seq 1 360); do
  runs="$(gh api \
    "repos/${GITHUB_REPOSITORY}/actions/workflows/${workflow}/runs?head_sha=${GITHUB_SHA}&per_page=20")"
  conclusion="$(printf '%s' "$runs" | jq -r '
    [.workflow_runs[] | select(.event == "push" or .event == "workflow_dispatch")]
    | sort_by(.created_at) | last | .conclusion // empty')"
  status="$(printf '%s' "$runs" | jq -r '
    [.workflow_runs[] | select(.event == "push" or .event == "workflow_dispatch")]
    | sort_by(.created_at) | last | .status // empty')"
  if [ "$status" = "completed" ]; then
    if [ "$conclusion" = "success" ]; then
      echo "${workflow} accepted exact source ${GITHUB_SHA}."
      exit 0
    fi
    echo "${workflow} rejected exact source ${GITHUB_SHA}: ${conclusion:-missing conclusion}" >&2
    exit 1
  fi
  echo "Waiting for ${workflow} on ${GITHUB_SHA} (${attempt}/360; status=${status:-not-started})"
  sleep 30
done

echo "${workflow} did not accept exact source ${GITHUB_SHA} within three hours" >&2
exit 1
