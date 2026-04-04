#!/usr/bin/env bash
# End-to-end pipeline: test, train (local smoke), evaluate.
# Usage:
#   ./ml/scripts/run_e2e.sh                          # run all stages
#   ./ml/scripts/run_e2e.sh --skip-test               # skip tests
#   ./ml/scripts/run_e2e.sh --skip-train              # skip training
#   ./ml/scripts/run_e2e.sh --test-only               # alias for --skip-train
#   ./ml/scripts/run_e2e.sh --train-only              # alias for --skip-test
#   ./ml/scripts/run_e2e.sh --modal                   # train on Modal GPU
#   ./ml/scripts/run_e2e.sh --eval-only checkpoint.pt # evaluate a checkpoint

set -euo pipefail
cd "$(dirname "$0")/../.."

SKIP_TEST=false
SKIP_TRAIN=false
USE_MODAL=false
EVAL_ONLY=false
EVAL_CHECKPOINT=""

# Parse flags
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-test)  SKIP_TEST=true; shift ;;
        --skip-train) SKIP_TRAIN=true; shift ;;
        --test-only)  SKIP_TRAIN=true; shift ;;
        --train-only) SKIP_TEST=true; shift ;;
        --modal)      USE_MODAL=true; shift ;;
        --eval-only)  EVAL_ONLY=true; shift; EVAL_CHECKPOINT="$1"; shift ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

# ─── Tests ────────────────────────────────────────────
run_tests() {
    echo "=== Running tests ==="
    uv run pytest ml/tests/ web/backend/tests/ -v
    echo ""
}

# ─── Local smoke train ────────────────────────────────
smoke_train() {
    echo "=== Smoke training (100 episodes, local) ==="
    uv run python -m guandan.training.train \
        --episodes 100 \
        --eval-interval 50 \
        --eval-games 10 \
        --batch-size 64 \
        --buffer-size 10000 \
        --train-steps 1 \
        --patience 99
    echo ""
}

# ─── Modal GPU train ─────────────────────────────────
modal_train() {
    echo "=== Launching Modal GPU training ==="
    modal run ml/scripts/modal/dmc.py
    echo ""
}

# ─── Evaluate checkpoint ─────────────────────────────
evaluate() {
    local checkpoint="$1"
    echo "=== Evaluating $checkpoint ==="
    uv run python ml/scripts/eval/checkpoint.py --checkpoint "$checkpoint" --opponent random --games 500
    echo ""
    uv run python ml/scripts/eval/checkpoint.py --checkpoint "$checkpoint" --opponent heuristic --games 500
    echo ""
}

# ─── Main ─────────────────────────────────────────────
if $EVAL_ONLY; then
    evaluate "$EVAL_CHECKPOINT"
    exit 0
fi

if ! $SKIP_TEST; then
    run_tests
fi

if ! $SKIP_TRAIN; then
    if $USE_MODAL; then
        modal_train
    else
        smoke_train
    fi
fi
