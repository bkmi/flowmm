#!/bin/bash

#SBATCH --error=prerelax/slurm/eval-%A_%a.err
#SBATCH --output=prerelax/slurm/eval-%A_%a.out
#SBATCH --job-name=pc
#SBATCH --mem=10
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --open-mode=append
#SBATCH --account=ocp
#SBATCH --qos=ami_shared
#SBATCH --signal=USR2@90
#SBATCH --time=180

export PROJECT_ROOT="/fsx-open-catalyst/bkmi/flowmm"
export HYDRA_JOBS="/fsx-open-catalyst/bkmi/flowmm"
export WABDB_DIR="/fsx-open-catalyst/bkmi/flowmm"

# ls /fsx-open-catalyst/bkmi/flowmm/results/2025.02.26/065840/*/*/consolidated_rfm-from-llm.pt | while read -r consolidated_pt; do
#     srun python scripts_eval/collected_to_cif.py ${consolidated_pt}
# done

# wait

ls /fsx-open-catalyst/bkmi/flowmm/results/2025.02.26/065840/*/*/rfm_outputs.csv | while read -r input_csv; do
    /home/bkmi/micromamba/envs/fairchem_flowmm/bin/python scripts_eval/prerelax_fc.py ${input_csv} --batch_size 8 --num_jobs 800 &
    sleep 300
done

wait
