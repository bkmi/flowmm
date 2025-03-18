#!/bin/bash

#SBATCH --cpus-per-task=70
#SBATCH --error=slurm/%x/eval-%A_%a.err
#SBATCH --output=slurm/%x/eval-%A_%a.out
#SBATCH --gpus-per-node=0
#SBATCH --job-name=split
#SBATCH --mem=1000GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --open-mode=append
#SBATCH --account=ocp
#SBATCH --qos=ami_shared
#SBATCH --signal=USR2@90
#SBATCH --time=1-0

CMD=`cat $CMDS_FILE | head -n ${SLURM_ARRAY_TASK_ID} | tail -n 1`
echo $CMD
$CMD
