#!/bin/bash
#SBATCH --job-name=sf-diag
#SBATCH --output=slurm/log/sf-diag-%j.txt
#SBATCH --account=crescendo
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --time=0-00:20:00
#SBATCH --chdir=/e/project1/crescendo/reim1/ml-simplefold
#
# DIAGNOSTIC ONLY: fresh (no resume) 200-step run to get the `simple` profiler
# report, which prints on teardown. Mirrors run_simplefold.sh's settings so the
# [get_train_batch] (dataloader wait) vs [run_training_batch] (compute) ratio is
# representative of the real run. Overrides: no checkpoint, max_steps=200.
#   sbatch slurm/diag_profile.sh
# Then read the profiler table near the END of slurm/log/sf-diag-<jobid>.txt.
#
set -euo pipefail

module purge
module use /e/project1/crescendo/hoffbauer1/easybuild/easybuild/jupiter/modules/all/Core
module load CUDA/12.8.0 GCC

export UV_CACHE_DIR=/e/project1/crescendo/reim1/uv-cache
export TRITON_HOME=/e/project1/crescendo/reim1/triton
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=WARN
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=1
mkdir -p "$UV_CACHE_DIR" "$TRITON_HOME"

source /e/project1/crescendo/reim1/ml-simplefold/.venv/bin/activate

srun python /e/project1/crescendo/reim1/ml-simplefold/src/simplefold/train.py \
  experiment=train \
  load_ckpt_path=null \
  trainer.max_steps=200
