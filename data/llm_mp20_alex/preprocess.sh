#!/bin/bash
inds=(
    0
    77565
    155131
    232696
    310261
    387826
    465392
    542957
    620522
    698087
    775653
    853218
    930783
    1008349
    1085914
    1163479
    1241044
    1318610
    1396175
    1473740
    1551305
    1628871
    1706436
    1784001
    1861567
    1939132
    2016697
    2094262
    2171828
    2249393
    2326958
    2404523
    2482089
    2559654
    2637219
    2714785
    2792350
    2869915
    2947480
    3025046
    3102611
    3180176
    3257741
    3335307
    3412872
    3490437
    3568003
    3645568
    3723133
    3800698
    3878264
    3955829
    4033394
    4110959
    4188525
    4266090
    4343655
    4421221
    4498786
    4576351
    4653916
    4731482
    4809047
    4886612
    4964177
    5041743
    5119308
    5196873
    5274439
    5352004
    5429569
    5507134
    5584700
    5662265
    5739830
    5817395
    5894961
    5972526
    6050091
    6127657
    6205222
    6282787
    6360352
    6437918
    6515483
    6593048
    6670613
    6748179
    6825744
    6903309
    6980875
    7058440
    7136005
    7213570
    7291136
    7368701
    7446266
    7523831
    7601397
    7678965
)
max_length=${#inds[@]}
max_length=$(echo "l($max_length)/l(10)" | bc -l)
max_length=$(echo "scale=0; $max_length/1" | bc -l)
max_length=$((max_length + 1))

CMDS_FILE=/home/bkmi/flowmm/data/llm_mp20_alex/CMDS_PREPROCESS.sh

for ((i=0; i<${#inds[@]}-1; i++)); do
    start=${inds[i]}
    end_repeat=${inds[i+1]}
    end=$((end_repeat - 1))
    # end=$(( ${inds[i+1]} - 1 ))
    padded_index=$(printf "%0${max_length}d" $i)

    # echo "Processing chunk: [$start, $end]"
    CMD="/home/bkmi/micromamba/envs/flowmm/bin/python \
        /home/bkmi/flowmm/src/flowmm/common/manual_preprocess.py \
        /home/bkmi/flowmm/data/llm_mp20_alex/train.csv \
        /home/bkmi/flowmm/data/llm_mp20_alex/train${padded_index}.pkl \
        --workers 70 \
        --start_ind ${start} \
        --end_ind ${end}"
    echo $CMD
done > $CMDS_FILE
NUM_JOBS=$(wc -l < $CMDS_FILE)
echo Submitting $NUM_JOBS jobs
CMDS_FILE=$CMDS_FILE sbatch --requeue --export CMDS_FILE --array=1-"$NUM_JOBS" /home/bkmi/flowmm/data/llm_mp20_alex/eval_slurm.sh
