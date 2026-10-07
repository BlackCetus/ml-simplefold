#!/bin/bash
#SBATCH --job-name=simplefold
#SBATCH --output=slurm/log/simplefold-%j.txt
#SBATCH --account=crescendo
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4          
#SBATCH --time=0-00:30:00
#SBATCH --chdir=/e/project1/crescendo/reim1/ml-simplefold

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
mkdir -p "$UV_CACHE_DIR" "$TRITON_HOME"

source /e/project1/crescendo/reim1/ml-simplefold/.venv/bin/activate

srun python /e/project1/crescendo/reim1/ml-simplefold/src/simplefold/train.py experiment=train