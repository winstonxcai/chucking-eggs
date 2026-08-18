#!/usr/bin/env bash
# Submit from the repository root. Slurm resource directives can be overridden
# at submission time, for example:
#
#   sbatch --cpus-per-task=32 --gres=gpu:1 \
#     --export=ALL,DART_UPDATES=50000 \
#     ml/scripts/slurm/dart_train.sbatch
#
# For a named GPU resource, use the GRES name configured by the cluster, e.g.
# --gres=gpu:a800:1. DART currently uses one CUDA learner, so request one GPU.
#SBATCH --job-name=dart-train
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=11:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --output=ml/runs/slurm-%x-%j.out

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

DART_CONDA_BASE="${DART_CONDA_BASE:-$HOME/miniconda3}"
source "$DART_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "${DART_CONDA_ENV:-dart-a800}"

# Runtime settings may be supplied with sbatch --export=ALL,NAME=value,...
DART_UPDATES="${DART_UPDATES:-50000}"
DART_ACTORS="${DART_ACTORS:-$SLURM_CPUS_PER_TASK}"
DART_CONFIG="${DART_CONFIG:-ml/src/guandan/dart/configs/dart_a800.yaml}"
DART_SEED="${DART_SEED:-0}"
DART_RUN_NAME="${DART_RUN_NAME:-dart_${SLURM_JOB_ID}_${DART_UPDATES}upd}"
DART_RUN_DIR="${DART_RUN_DIR:-$SLURM_SUBMIT_DIR/ml/runs/$DART_RUN_NAME}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export DART_TQDM=0
export DART_STREAM_LOGS=1
# Some Slurm nodes isolate POSIX named semaphores in a way that breaks Python's
# spawn context while actors are being created. Fork is safe here because the
# parent has not created a CUDA context; the learner initializes CUDA itself.
export DART_MP_START_METHOD="${DART_MP_START_METHOD:-fork}"

DART_JOB_SCRATCH="${DART_JOB_SCRATCH:-${SLURM_TMPDIR:-${TMPDIR:-/tmp}/${USER}/dart-${SLURM_JOB_ID}}}"
export DART_WEIGHT_DIR="${DART_WEIGHT_DIR:-$DART_JOB_SCRATCH/weights}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$DART_JOB_SCRATCH/torchinductor}"

mkdir -p "$DART_RUN_DIR"
mkdir -p "$DART_WEIGHT_DIR"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR"

echo "DART Slurm training"
echo "  job:     $SLURM_JOB_ID"
echo "  CPUs:    $SLURM_CPUS_PER_TASK"
echo "  actors:  $DART_ACTORS"
echo "  updates: $DART_UPDATES"
echo "  config:  $DART_CONFIG"
echo "  run dir: $DART_RUN_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

TRAIN_ARGS=(
  --config "$DART_CONFIG"
  --n-actors "$DART_ACTORS"
  --updates "$DART_UPDATES"
  --device cuda
  --seed "$DART_SEED"
  --run-dir "$DART_RUN_DIR"
)

if [[ -n "${DART_RESUME:-}" ]]; then
  TRAIN_ARGS+=(--resume "$DART_RESUME")
fi

exec srun \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task="$SLURM_CPUS_PER_TASK" \
  --cpu-bind=cores \
  python -m guandan.dart "${TRAIN_ARGS[@]}" "$@"
