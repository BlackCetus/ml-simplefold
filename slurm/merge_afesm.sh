#!/bin/bash
#SBATCH --job-name=afesm-merge
#SBATCH --output=slurm/log/afesm-merge-%j.txt
#SBATCH --account=crescendo
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4               # booster forces GPUs; merge is a ~few-min symlink+concat step
#SBATCH --time=0-00:30:00
#SBATCH --chdir=/e/project1/crescendo/reim1/ml-simplefold
#
# Merge array-task parts (OUT/parts/*/wds) into OUT/wds and finalize.
# Env: OUT (required), CLUSTER_SRC (optional, e.g. afesme_dict.json -> filtered_afesm_cluster.json)
#
set -euo pipefail
module purge
module use /e/project1/crescendo/hoffbauer1/easybuild/easybuild/jupiter/modules/all/Core
module load CUDA/12.8.0 GCC
export UV_CACHE_DIR=/e/project1/crescendo/reim1/uv-cache
source /e/project1/crescendo/reim1/ml-simplefold/.venv/bin/activate

: "${OUT:?set OUT}"
echo "[merge] OUT=$OUT CLUSTER_SRC=${CLUSTER_SRC:-<none>}"
python data/afesm_stage.py --merge --out "$OUT" ${CLUSTER_SRC:+--cluster-src "$CLUSTER_SRC"}
echo "[merge] DONE"
