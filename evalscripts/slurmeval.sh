#!/bin/bash

#SBATCH --error=hps/slurm/eval-%A_%a.err
#SBATCH --output=hps/slurm/eval-%A_%a.out
#SBATCH --gres=gpu:8
#SBATCH --job-name=eval_rfm
#SBATCH --mem=0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --open-mode=append
#SBATCH --account=ocp
#SBATCH --qos=ami_shared
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
# CMD="${CMD#*&&}"
# CMD2="${CMD%%&&*}"

srun --ntasks 8 --nodes 1 ${CMD0}
srun --ntasks 1 --nodes 1 --exclusive ${CMD1}
