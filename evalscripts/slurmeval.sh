#!/bin/bash

#SBATCH --cpus-per-task=10
#SBATCH --error=hps/slurm/eval-%A_%a.err
#SBATCH --output=hps/slurm/eval-%A_%a.out
#SBATCH --gpus-per-node=1
#SBATCH --job-name=eval_rfm
#SBATCH --mem=10GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --open-mode=append
#SBATCH --partition=ocp
#SBATCH --signal=USR2@90
#SBATCH --time=7200

export PROJECT_ROOT="/private/home/bmwood/ocp_results/rfm_gen/solids"
export HYDRA_JOBS="/private/home/bmwood/ocp_results/rfm_gen/solids"
export WANDB_DIR="/private/home/bmwood/ocp_results/rfm_gen/solids"

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
