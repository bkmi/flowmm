#!/bin/bash
pickles=(/home/bkmi/flowmm/data/llm_osc/train*.pkl)
python /home/bkmi/flowmm/src/flowmm/common/combine_manual_preprocessed.py /home/bkmi/flowmm/data/llm_osc/train_ori.pt --pickles ${pickles[@]}
python /home/bkmi/flowmm/src/flowmm/common/combine_manual_preprocessed.py /home/bkmi/flowmm/data/llm_osc/val_ori.pt --pickles /home/bkmi/flowmm/data/llm_osc/val_t1.5.pkl
