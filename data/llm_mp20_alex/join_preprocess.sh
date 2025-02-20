#!/bin/bash
pickles=(/home/bkmi/flowmm/data/llm_mp20_alex/*.pkl)
train=("${pickles[@]:0:98}")
val=("${pickles[@]:98}")
python /home/bkmi/flowmm/src/flowmm/common/combine_manual_preprocessed.py /home/bkmi/flowmm/data/llm_mp20_alex/train_ori.pt --pickles ${train[@]}
python /home/bkmi/flowmm/src/flowmm/common/combine_manual_preprocessed.py /home/bkmi/flowmm/data/llm_mp20_alex/val_ori.pt --pickles ${val[@]}
