#!/usr/bin/env bash
# Submit one 10k CPU/actor/lane throughput probe. Override CPU allocation at
# submission time, for example: sbatch --cpus-per-task=84 ... this-file.
#SBATCH --job-name=dart-a800-probe
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --output=ml/runs/slurm-%x-%j.out

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

DART_CONDA_BASE="${DART_CONDA_BASE:-$HOME/miniconda3}"
source "$DART_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "${DART_CONDA_ENV:-dart-a800}"

DART_ACTORS="${DART_ACTORS:-$SLURM_CPUS_PER_TASK}"
DART_LANES="${DART_LANES:-32}"
DART_PROBE_UPDATES="${DART_PROBE_UPDATES:-10000}"
DART_WARMUP_UPDATES="${DART_WARMUP_UPDATES:-500}"
DART_SWEEP_STAGE="${DART_SWEEP_STAGE:-cpu_scaling}"
DART_SWEEP_NAME="${DART_SWEEP_NAME:-cpu${SLURM_CPUS_PER_TASK}_actors${DART_ACTORS}_lanes${DART_LANES}}"
DART_SWEEP_ROOT="${DART_SWEEP_ROOT:-$SLURM_SUBMIT_DIR/ml/runs/a800_cpu_lane_${DART_SWEEP_NAME}_${SLURM_JOB_ID}}"
DART_SWEEP_SCRATCH="${DART_SWEEP_SCRATCH:-${TMPDIR:-/tmp}/${USER}/a800-cpu-lane-${SLURM_JOB_ID}}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export DART_TQDM=0
export DART_STREAM_LOGS=1
export DART_LEARNER_MP_START_METHOD="${DART_LEARNER_MP_START_METHOD:-spawn}"
export DART_ACTOR_MP_START_METHOD="${DART_ACTOR_MP_START_METHOD:-fork}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$DART_SWEEP_SCRATCH/torchinductor}"

mkdir -p "$DART_SWEEP_ROOT" "$DART_SWEEP_SCRATCH" "$TORCHINDUCTOR_CACHE_DIR"

exec srun \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task="$SLURM_CPUS_PER_TASK" \
  --cpu-bind=cores \
  python ml/scripts/util/run_a800_throughput_probe.py \
  --output-root "$DART_SWEEP_ROOT" \
  --scratch-root "$DART_SWEEP_SCRATCH" \
  --name "$DART_SWEEP_NAME" \
  --stage "$DART_SWEEP_STAGE" \
  --actors "$DART_ACTORS" \
  --lanes "$DART_LANES" \
  --updates "$DART_PROBE_UPDATES" \
  --warmup-updates "$DART_WARMUP_UPDATES"
