#!/bin/bash

inds=(
    0
    26119
    52238
    78357
    104476
    130595
    156714
    182832
    208951
    235070
    261189
    287308
    313427
    339546
    365665
    391784
    417903
    444022
    470141
    496259
    522378
    548497
    574616
    600735
    626854
    652973
    679092
    705211
    731330
    757449
    783568
    809686
    835805
    861924
    888043
    914162
    940281
    966400
    992519
    1018638
    1044757
    1070876
    1096995
    1123113
    1149232
    1175351
    1201470
    1227589
    1253708
    1279829  # must increase this by a little from the actual amount
)

# get the max length for padding the index
max_length=${#inds[@]}
max_length=$(echo "l($max_length)/l(10)" | bc -l)
max_length=$(echo "scale=0; $max_length/1" | bc -l)
max_length=$((max_length + 1))

CMDS_FILE=/home/bkmi/flowmm/data/llm_mp20_alex_abridged/CMDS_PREPROCESS.sh

for ((i=0; i<${#inds[@]}-1; i++)); do
    start=${inds[i]}
    end_repeat=${inds[i+1]}
    end=$((end_repeat - 1))
    padded_index=$(printf "%0${max_length}d" $i)

    # echo "Processing chunk: [$start, $end]"
    CMD="/home/bkmi/micromamba/envs/flowmm/bin/python \
        /home/bkmi/flowmm/src/flowmm/common/manual_preprocess.py \
        /home/bkmi/flowmm/data/llm_mp20_alex_abridged/train.csv \
        /home/bkmi/flowmm/data/llm_mp20_alex_abridged/train${padded_index}.pkl \
        --workers 70 \
        --start_ind ${start} \
        --end_ind ${end}"
    echo $CMD
done > $CMDS_FILE

# validation
CMD="/home/bkmi/micromamba/envs/flowmm/bin/python \
    /home/bkmi/flowmm/src/flowmm/common/manual_preprocess.py \
    /home/bkmi/flowmm/data/llm_mp20_alex_abridged/val.csv \
    /home/bkmi/flowmm/data/llm_mp20_alex_abridged/val.pkl \
    --workers 70"
echo $CMD >> $CMDS_FILE

NUM_JOBS=$(wc -l < $CMDS_FILE)
echo Submitting $NUM_JOBS jobs
CMDS_FILE=$CMDS_FILE sbatch --requeue --export CMDS_FILE --array=1-"$NUM_JOBS" /home/bkmi/flowmm/data/llm_mp20_alex_abridged/eval_slurm.sh
