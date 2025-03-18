#!/bin/bash
inds=(
    0
    22438
    44876
    67314
    89752
    112190
    134628
    157066
    179504
    201943
    224381
    246819
    269257
    291695
    314133
    336571
    359009
    381447
    403885
    426323
    448761
    471199
    493637
    516075
    538513
    560952
    583390
    605828
    628266
    650704
    673142
    695580
    718018
    740456
    762894
    785332
    807770
    830208
    852646
    875084
    897522
    919961
    942399
    964837
    987275
    1009713
    1032151
    1054589
    1077027
    1099467  # must increase this by a little from the actual amount
)
max_length=${#inds[@]}
max_length=$(echo "l($max_length)/l(10)" | bc -l)
max_length=$(echo "scale=0; $max_length/1" | bc -l)
max_length=$((max_length + 1))

CMDS_FILE=/fsx-open-catalyst/bkmi/flowmm/data/llm_osc/CMDS_PREPROCESS.sh

for ((i=0; i<${#inds[@]}-1; i++)); do
    start=${inds[i]}
    end_repeat=${inds[i+1]}
    end=$((end_repeat - 1))
    # end=$(( ${inds[i+1]} - 1 ))
    padded_index=$(printf "%0${max_length}d" $i)

    # echo "Processing chunk: [$start, $end]"
    CMD="/home/bkmi/micromamba/envs/flowmm/bin/python \
        /fsx-open-catalyst/bkmi/flowmm/scripts_dataprep/manual_preprocess.py \
        /fsx-open-catalyst/bkmi/flowmm/data/llm_osc/train_t1.5.csv \
        /fsx-open-catalyst/bkmi/flowmm/data/llm_osc/train_t1.5_${padded_index}.pkl \
        --workers 70 \
        --start_ind ${start} \
        --end_ind ${end}"
    echo $CMD
done > $CMDS_FILE

# validation
CMD="/home/bkmi/micromamba/envs/flowmm/bin/python \
    /fsx-open-catalyst/bkmi/flowmm/scripts_dataprep/manual_preprocess.py \
    /fsx-open-catalyst/bkmi/flowmm/data/llm_osc/val_t1.5.csv \
    /fsx-open-catalyst/bkmi/flowmm/data/llm_osc/val_t1.5.pkl \
    --workers 70"
echo $CMD >> $CMDS_FILE

NUM_JOBS=$(wc -l < $CMDS_FILE)
echo Submitting $NUM_JOBS jobs
CMDS_FILE=$CMDS_FILE sbatch --requeue --export CMDS_FILE --array=1-"$NUM_JOBS" /fsx-open-catalyst/bkmi/flowmm/data/llm_osc/eval_slurm.sh
