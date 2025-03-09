export CKPT_ROOT="/fsx-open-catalyst/bkmi/flowmm/"
export PMG_VASP_PSP_DIR="/checkpoint/zulissi/mongo_fw_atomate_config/pps"
export PROJECT_ROOT="/fsx-open-catalyst/bkmi/flowmm"
export HYDRA_JOBS="/fsx-open-catalyst/bkmi/flowmm"
export WABDB_DIR="/fsx-open-catalyst/bkmi/flowmm"

CMDS_FILE=/fsx-open-catalyst/bkmi/flowmm/evalscripts/CMDS_EVAL.sh
root=/fsx-open-catalyst/bkmi/flowmm/results/2025.02.26/065840

llm_sample=/fsx-open-catalyst/bkmi/flowmm/data/samples_t1.5_merged.csv
numsteps=250

/home/bkmi/micromamba/envs/flowmm/bin/python /fsx-open-catalyst/bkmi/flowmm/evalscripts/highestepochfromhydrasweep.py ${root} | while read -r ckpt; do
    # create directory for evals
    stem=${ckpt##*/}  # retain part after the last slash
    stem=${stem%.*}  # retain part before the period
    subdir="../stem=${stem}_numsteps=${numsteps}"

    # write commands
    CMD_GEN="/home/bkmi/micromamba/envs/flowmm/bin/python scripts_model/evaluate.py \
        rfm-from-llm \
        ${ckpt} \
        ${llm_sample} \
        --subdir ${subdir} \
        --num_steps ${numsteps} \
        --batch_size 8192 \
        --multi_gpu"
    CMD_CONS="/home/bkmi/micromamba/envs/flowmm/bin/python scripts_model/evaluate.py \
        consolidate \
        ${ckpt} \
        --subdir ${subdir}"
    CMD="$CMD_GEN && $CMD_CONS"
    echo $CMD
done > $CMDS_FILE

NUM_JOBS=$(wc -l < $CMDS_FILE)
echo Submitting $NUM_JOBS jobs
CMDS_FILE=$CMDS_FILE sbatch --requeue --export CMDS_FILE --array=1-"$NUM_JOBS" /fsx-open-catalyst/bkmi/flowmm/evalscripts/slurmeval.sh
