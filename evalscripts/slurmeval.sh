#!/bin/bash

#SBATCH --cpus-per-task=10
#SBATCH --error=slurm/eval-%A_%a.err
#SBATCH --output=slurm/eval-%A_%a.out
#SBATCH --gpus-per-node=1
#SBATCH --job-name=eval_rfm
#SBATCH --mem=10GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --open-mode=append
#SBATCH --partition=ocp
#SBATCH --signal=USR2@90
#SBATCH --time=7200

export PROJECT_ROOT="/fsx-open-catalyst/bkmi/flowmm"
export HYDRA_JOBS="/fsx-open-catalyst/bkmi/flowmm"
export WABDB_DIR="/fsx-open-catalyst/bkmi/flowmm"

CMD=`cat $CMDS_FILE | head -n ${SLURM_ARRAY_TASK_ID} | tail -n 1`
echo $CMD

# `$CMD`

# Split Command
CMD0="${CMD%%&&*}"
CMD="${CMD#*&&}"
CMD1="${CMD%%&&*}"
CMD="${CMD#*&&}"
CMD2="${CMD%%&&*}"

$CMD0 && $CMD1 && $CMD2
