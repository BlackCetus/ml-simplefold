#!/bin/bash
#SBATCH --job-name=afesm-stage-arr
#SBATCH --output=slurm/log/afesm-stage-%A_%a.txt
#SBATCH --account=crescendo
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:4               # booster is GPU-only; staging is CPU-bound (GPUs idle) — array spreads it over nodes
#SBATCH --time=0-12:00:00
#SBATCH --chdir=/e/project1/crescendo/reim1/ml-simplefold
#
# ARRAY staging: each task stages a disjoint 1/N slice of ID_LIST into its own
# OUT/parts/<PREFIX>_<taskid> subdir (no shared files -> no races). Launch with
# --array to set N, then run ONE merge afterwards (see below).
#
# Generic over source — pass ID_LIST / DB / PREFIX / OUT:
#   sbatch --array=0-19 --export=ALL,\
#ID_LIST=data/afesm_e/afdb_ids.foldcomp.txt,DB=/e/scratch/crescendo/foldcomp_dbs/afdb_uniprot_v4,PREFIX=afdb,OUT=/e/fscratch/crescendo/afesm_stage \
#     slurm/stage_afesm_array.sh
#
# After ALL array tasks finish (all sources into the same OUT), merge + cluster file
# ON A LOGIN NODE (light, no GPU, sees /e):
#   .venv/bin/python data/afesm_stage.py --merge --out $OUT \
#       --cluster-src data/afesm_e/afesme_dict.json
#
set -euo pipefail
module purge
module use /e/project1/crescendo/hoffbauer1/easybuild/easybuild/jupiter/modules/all/Core
module load CUDA/12.8.0 GCC
export OMP_NUM_THREADS=1
export UV_CACHE_DIR=/e/project1/crescendo/reim1/uv-cache
source /e/project1/crescendo/reim1/ml-simplefold/.venv/bin/activate

: "${ID_LIST:?set ID_LIST}"; : "${DB:?set DB}"; : "${PREFIX:?set PREFIX}"; : "${OUT:?set OUT}"
SHARD_SIZE=${SHARD_SIZE:-15000}   # ~1.5GB tars (AFDB ~13k, ESM ~19k /shard); override per source
JOBS=${JOBS:-$(( ${SLURM_CPUS_ON_NODE:-$(nproc)} - 4 ))}
N=${SLURM_ARRAY_TASK_COUNT:-1}
I=${SLURM_ARRAY_TASK_ID:-0}

# CRITICAL: per-shard transient files (50k pdb/cif/npz/json/pkl) must live on
# NODE-LOCAL storage, NOT on the shared fscratch — otherwise all concurrent tasks
# pile millions of inodes onto one GPFS quota and blow it. Only the final tars
# (a handful of inodes) go to OUT on fscratch.
# pick the node-local dir with the MOST free space (don't assume /dev/shm size)
LOCAL=""; BEST=0
for d in "${SLURM_TMPDIR:-}" /dev/shm /tmp; do
  [ -n "$d" ] && [ -d "$d" ] && [ -w "$d" ] || continue
  free=$(df -Pk "$d" 2>/dev/null | awk 'NR==2{print int($4)}')   # KiB
  [ -n "$free" ] && [ "$free" -gt "$BEST" ] && { BEST=$free; LOCAL=$d; }
done
GIB=$(( BEST / 1024 / 1024 ))
[ -z "$LOCAL" ] && { echo "ERROR: no node-local scratch found"; exit 1; }
# transient peak ~= SHARD_SIZE * 418KB (cif+npz+pkl coexisting); require ~2x margin
NEED=$(( SHARD_SIZE * 418 * 2 / 1024 / 1024 )); [ "$NEED" -lt 4 ] && NEED=4
if [ "$GIB" -lt "$NEED" ]; then
  echo "ERROR: node-local $LOCAL has ${GIB}GiB free, need ~${NEED}GiB for shard-size $SHARD_SIZE; lower SHARD_SIZE"; exit 1
fi
WORK="$LOCAL/afesm_work_${SLURM_JOB_ID}_${I}"
mkdir -p "$WORK"
trap 'rm -rf "$WORK"' EXIT
echo "[stage-arr] node-local WORK=$WORK on $LOCAL (${GIB}GiB free)"

echo "[stage-arr] part $I/$N  PREFIX=$PREFIX ID_LIST=$ID_LIST DB=$DB OUT=$OUT JOBS=$JOBS SHARD_SIZE=$SHARD_SIZE"
python data/afesm_stage.py \
  --id-list "$ID_LIST" --db "$DB" --out "$OUT" --work "$WORK" \
  --shard-prefix "$PREFIX" --part "$I" --num-parts "$N" \
  --shard-size "$SHARD_SIZE" --pack webdataset -j "$JOBS"
echo "[stage-arr] part $I/$N DONE"
