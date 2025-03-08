export CKPT_ROOT="/checkpoint/bkmi/solids/"
export PMG_VASP_PSP_DIR="/checkpoint/zulissi/mongo_fw_atomate_config/pps"
export PROJECT_ROOT="/private/home/bkmi/sci/solids"
export HYDRA_JOBS="/private/home/bkmi/sci/solids"
export WABDB_DIR="/private/home/bkmi/sci/solids"

CMDS_FILE=evalscripts/CMDS_EVAL.sh
root=/home/bkmi/flowmm/results/2025.02.26/065840

python /home/bkmi/flowmm/evalscripts/highestepochfromhydrasweep.py ${root} | while read -r ckpt; do
    run_root=$(dirname "$(dirname "${ckpt}")")
    numsteps=100

    # create directory for evals
    stem=${ckpt##*/}  # retain part after the last slash
    stem=${stem%.*}  # retain part before the period
    subdir="stem=${stem}_numsteps=${numsteps}"

    llm_sample=data/samples_1.csv  # TODO, num ranks?

    # EXAMPLE
    # python scripts_model/evaluate.py rfm-from-llm /home/bkmi/flowmm/results/2025.02.26/065840/0/checkpoints/epoch=79-step=806240.ckpt /home/bkmi/flowmm/data/anuroop_llm_mp20_alex/samples_t1.5/samples_1.csv

    CMD_GEN="python scripts_model/evaluate.py \
        rfm-from-llm \
        ${base} \
        ${llm_sample} \
        --subdir ${subdir} \
        --batch_size 1_000 \
        --rfm_from_llm_id ${rank} \
        --num_steps ${numsteps}"
    CMD_CONS="python scripts_model/evaluate.py \
        consolidate ${base} \
        --subdir ${subdir}"
    CMD_MET="python scripts_model/evaluate.py old_eval_metrics ${base} --subdir ${subdir}"

    CMD="$CMD_GEN && $CMD_CONS && $CMD_MET"
    echo $CMD

    python scripts_model/evaluate.py rfm-from-llm /Users/bkmi/Documents/flowmm/0/checkpoints/epoch=69-step=705460.ckpt /Users/bkmi/Documents/flowmm/data/samples_1.csv
done


# ls hps/unconditional-mp_20/2024-01-25/*/*/every_n_epochs -d | while read -r parent; do
python /home/bkmi/flowmm/evalscripts/highestepochfromhydrasweep.py | while read -r parent; do
    ls "$parent" | grep -E "499|899" | grep .ckpt | while read -r ckpt ; do
        base="$parent"/"$ckpt"

        stem=${base##*/}  # retain part after the last slash
        stem=${stem%.*}  # retain part before the period

        for numsteps in 500 1_000; do
            subdir="stem=${stem}_numsteps=${numsteps}"

            CMD_GEN="/private/home/bkmi/mambaforge/envs/solids2/bin/python scripts_model/evaluate.py generate ${base} --subdir ${subdir} --num_samples 2_000 --batch_size 1_000 --num_steps ${numsteps}"
            CMD_CONS="/private/home/bkmi/mambaforge/envs/solids2/bin/python scripts_model/evaluate.py consolidate ${base} --subdir ${subdir}"
            CMD_MET="/private/home/bkmi/mambaforge/envs/solids2/bin/python scripts_model/evaluate.py old_eval_metrics ${base} --subdir ${subdir}"

            CMD="$CMD_GEN && $CMD_CONS && $CMD_MET"
            echo $CMD
        done
    done
done > $CMDS_FILE

# NUM_JOBS=$(wc -l < $CMDS_FILE)
# echo Submitting $NUM_JOBS jobs
# CMDS_FILE=$CMDS_FILE sbatch --requeue --export CMDS_FILE --array=1-"$NUM_JOBS" eval_slurm.sh


# CMDS_FILE=CMDS_EVAL_uncond_perov.sh

# ls hps/unconditional-perov/2024-01-25/*/*/every_n_epochs -d | while read -r parent; do
#     ls "$parent" | grep -E "999" | grep .ckpt | while read -r ckpt ; do
#         base="$parent"/"$ckpt"

#         stem=${base##*/}  # retain part after the last slash
#         stem=${stem%.*}  # retain part before the period

#         for numsteps in 1_000; do
#             subdir="stem=${stem}_numsteps=${numsteps}"

#             CMD_GEN="/private/home/bkmi/mambaforge/envs/solids2/bin/python scripts_model/evaluate.py generate ${base} --subdir ${subdir} --num_samples 2_000 --batch_size 1_000  --num_steps ${numsteps}"
#             CMD_CONS="/private/home/bkmi/mambaforge/envs/solids2/bin/python scripts_model/evaluate.py consolidate ${base} --subdir ${subdir}"
#             CMD_MET="/private/home/bkmi/mambaforge/envs/solids2/bin/python scripts_model/evaluate.py old_eval_metrics ${base} --subdir ${subdir}"

#             CMD="$CMD_GEN && $CMD_CONS && $CMD_MET"
#             echo $CMD
#         done
#     done
# done > $CMDS_FILE

# NUM_JOBS=$(wc -l < $CMDS_FILE)
# echo Submitting $NUM_JOBS jobs
# CMDS_FILE=$CMDS_FILE sbatch --requeue --export CMDS_FILE --array=1-"$NUM_JOBS" eval_slurm.sh
