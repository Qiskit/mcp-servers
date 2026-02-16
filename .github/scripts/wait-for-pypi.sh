#!/usr/bin/env bash
# Wait for a package version to become available on PyPI.
# Reads the package identifier and version from server.json in the
# current working directory.
#
# Usage: .github/scripts/wait-for-pypi.sh [max_attempts] [sleep_seconds]
#   max_attempts  – number of polling attempts (default: 30)
#   sleep_seconds – seconds between attempts  (default: 20)
#
# Total timeout = max_attempts × sleep_seconds (default: 10 minutes).

set -euo pipefail

MAX_ATTEMPTS="${1:-30}"
SLEEP_SECONDS="${2:-20}"

PACKAGE=$(python3 -c "import json; print(json.load(open('server.json'))['packages'][0]['identifier'])")
VERSION=$(python3 -c "import json; print(json.load(open('server.json'))['packages'][0]['version'])")

echo "Waiting for ${PACKAGE}==${VERSION} on PyPI..."

for i in $(seq 1 "$MAX_ATTEMPTS"); do
  if curl -sf "https://pypi.org/pypi/${PACKAGE}/${VERSION}/json" > /dev/null 2>&1; then
    echo "✓ ${PACKAGE}==${VERSION} is available on PyPI (attempt ${i})"
    exit 0
  fi
  echo "Attempt ${i}/${MAX_ATTEMPTS}: not yet available, waiting ${SLEEP_SECONDS}s..."
  sleep "$SLEEP_SECONDS"
done

echo "✗ Timed out waiting for ${PACKAGE}==${VERSION} on PyPI after $(( MAX_ATTEMPTS * SLEEP_SECONDS )) seconds"
exit 1
