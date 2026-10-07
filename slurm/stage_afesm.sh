#!/bin/bash
#SBATCH --job-name=afesm-stage
#SBATCH --output=slurm/log/afesm-stage-%j.txt
#SBATCH --account=crescendo
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # staging is ONE orchestrator process (not 4-rank DDP)
#SBATCH --cpus-per-task=72           # give it the whole node's cores; process_mmcif fans out
#SBATCH --gres=gpu:4                 # booster is GPU-only: these 4 GPUs sit IDLE during staging
#SBATCH --time=0-12:00:00            # resumable — if it times out, just resubmit (shards are skipped)
#SBATCH --chdir=/e/project1/crescendo/reim1/ml-simplefold
#
# Staging driver for the AFESM distillation set.  CPU-only, so it wastes the node's
# 4 GPUs — minimize that by (a) the high -j below and (b) not staging more than needed
# (see SET/HALF/LIMIT). The driver is resumable: completed shards are skipped on resubmit.
#
# Usage:
#   mkdir -p slurm/log
#   # cheap smoke slice first (use a devel/short partition if your site has one):
#   sbatch --time=0-00:30:00 --export=ALL,LIMIT=2000 slurm/stage_afesm.sh
#   # full AFESM-E run:
#   sbatch --export=ALL,SET=afesm_e slurm/stage_afesm.sh
#   # cheaper alternatives: HALF=afdb (skip ESM), or point --id-list at a reps-only list
#
set -euo pipefail
module purge
module use /e/project1/crescendo/hoffbauer1/easybuild/easybuild/jupiter/modules/all/Core
module load CUDA/12.8.0 GCC
export OMP_NUM_THREADS=1                     # avoid thread oversubscription across workers
export UV_CACHE_DIR=/e/project1/crescendo/reim1/uv-cache
source /e/project1/crescendo/reim1/ml-simplefold/.venv/bin/activate

# ---- parameters (override via --export=ALL,VAR=...) ------------------------------
DB=${DB:-/e/scratch/crescendo/foldcomp_dbs}          # foldcomp DBs on ExaSM (/e) — booster nodes DON'T mount /p
OUT=${OUT:-/e/fscratch/crescendo/afesm_stage}        # staged output (ExaSM flash, next to PDB)
SET=${SET:-afesm_e}                                  # data/<SET>/{afdb_ids.foldcomp.txt,esm_ids.hq.txt,afesme_dict.json}
SHARD_SIZE=${SHARD_SIZE:-50000}
LIMIT=${LIMIT:-0}                                    # 0 = all; small (e.g. 2000) for a smoke run
HALF=${HALF:-both}                                   # afdb | esm | both
JOBS=${JOBS:-$(( ${SLURM_CPUS_ON_NODE:-$(nproc)} - 4 ))}   # leave a few cores for redis + the main proc

echo "[stage] OUT=$OUT SET=$SET SHARD_SIZE=$SHARD_SIZE LIMIT=$LIMIT HALF=$HALF JOBS=$JOBS"
lim=(); [ "$LIMIT" -gt 0 ] && lim=(--limit "$LIMIT")

# ---- AFDB half (afdb_uniprot_v4, 100% covered) -----------------------------------
if [ "$HALF" = afdb ] || [ "$HALF" = both ]; then
  python data/afesm_stage.py \
    --id-list "data/$SET/afdb_ids.foldcomp.txt" \
    --db "$DB/afdb_uniprot_v4" --out "$OUT" --shard-prefix afdb \
    --shard-size "$SHARD_SIZE" --pack webdataset -j "$JOBS" "${lim[@]}"
fi

# ---- ESM half (highquality_clust30) + cluster file (generated on the final step) --
if [ "$HALF" = esm ] || [ "$HALF" = both ]; then
  python data/afesm_stage.py \
    --id-list "data/$SET/esm_ids.hq.txt" \
    --db "$DB/highquality_clust30" --out "$OUT" --shard-prefix esm \
    --shard-size "$SHARD_SIZE" --pack webdataset -j "$JOBS" "${lim[@]}" \
    --cluster-src "data/$SET/afesme_dict.json"
fi

# ---- if ESM was skipped, still (re)build the cluster file from the AFDB-only ids --
if [ "$HALF" = afdb ]; then
  python - "$OUT" "data/$SET/afesme_dict.json" <<'PY'
import sys; sys.path.insert(0, "data")
import afesm_stage as A
class _A: pass
a = _A(); a.out = sys.argv[1]; a.pack = "webdataset"; a.cluster_src = sys.argv[2]
A.write_filtered_cluster(a)
PY
fi
echo "[stage] DONE"
