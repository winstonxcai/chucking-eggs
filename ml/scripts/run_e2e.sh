#!/usr/bin/env bash
# Engine-correctness test entry point.
#
# All training and eval scripts were archived to ml/_archive/ on 2026-04-24.
# See ml/LOGBOOK.md for the chronology and ml/_archive/README.md for restore.
#
# Usage:
#   ./ml/scripts/run_e2e.sh                # run all tests
#   ./ml/scripts/run_e2e.sh --backend-only # web/backend tests only
#   ./ml/scripts/run_e2e.sh --engine-only  # ml engine tests only

set -euo pipefail
cd "$(dirname "$0")/../.."

case "${1:-}" in
    --backend-only) uv run pytest web/backend/tests/ -v ;;
    --engine-only)  uv run pytest ml/tests/ -v ;;
    *)              uv run pytest ml/tests/ web/backend/tests/ -v ;;
esac
