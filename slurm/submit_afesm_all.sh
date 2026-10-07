#!/bin/bash
# Submit the whole staging pipeline with auto-merge.  Run on a LOGIN node.
#
# Submits three array jobs (AFDB+ESM -> AF, SwissProt -> SP), then two merge jobs
# that fire automatically via --dependency=afterok once their arrays fully succeed.
#
#   bash slurm/submit_afesm_all.sh
#
set -euo pipefail
cd /e/project1/crescendo/reim1/ml-simplefold

DB=${DB:-/e/scratch/crescendo/foldcomp_dbs}
AF=${AF:-/e/fscratch/crescendo/afesm_stage}
SP=${SP:-/e/fscratch/crescendo/swissprot_stage}

# --parsable makes sbatch print just the job id (for array jobs, the array's id)
afdb=$(sbatch --parsable --array=0-23 \
  --export=ALL,ID_LIST=data/afesm_e/afdb_ids.foldcomp.txt,DB=$DB/afdb_uniprot_v4,PREFIX=afdb,OUT=$AF,SHARD_SIZE=13000 \
  slurm/stage_afesm_array.sh)
echo "AFDB array   = $afdb"

esm=$(sbatch --parsable --array=0-3 \
  --export=ALL,ID_LIST=data/afesm_e/esm_ids.hq.txt,DB=$DB/highquality_clust30,PREFIX=esm,OUT=$AF,SHARD_SIZE=19000 \
  slurm/stage_afesm_array.sh)
echo "ESM  array   = $esm"

sp=$(sbatch --parsable --array=0-3 \
  --export=ALL,ID_LIST=data/swissprot/swissprot_ids.foldcomp.txt,DB=$DB/afdb_swissprot_v4,PREFIX=sp,OUT=$SP,SHARD_SIZE=13000 \
  slurm/stage_afesm_array.sh)
echo "SP   array   = $sp"

# AFESM merge waits for BOTH afdb and esm arrays to succeed; builds the cluster file
afmerge=$(sbatch --parsable --dependency=afterok:$afdb:$esm \
  --export=ALL,OUT=$AF,CLUSTER_SRC=data/afesm_e/afesme_dict.json \
  slurm/merge_afesm.sh)
echo "AFESM merge  = $afmerge  (afterok:$afdb:$esm)"

spmerge=$(sbatch --parsable --dependency=afterok:$sp \
  --export=ALL,OUT=$SP \
  slurm/merge_afesm.sh)
echo "SP merge     = $spmerge  (afterok:$sp)"

echo
echo "Submitted. Watch:  squeue --me"
echo "NOTE: afterok fires only if EVERY array task succeeds. If a task times out"
echo "(12h wall), Slurm cancels the merge (DependencyNeverSatisfied). Then resubmit"
echo "the timed-out array (it resumes/skips done shards) and rerun the merge, e.g.:"
echo "  sbatch --export=ALL,OUT=$AF,CLUSTER_SRC=data/afesm_e/afesme_dict.json slurm/merge_afesm.sh"
