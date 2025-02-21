#!/bin/bash
pickles=(/home/bkmi/flowmm/data/llm_mp20_alex_abridged/*.pkl)
python /home/bkmi/flowmm/src/flowmm/common/combine_manual_preprocessed.py /home/bkmi/flowmm/data/llm_mp20_alex_abridged/train_ori.pt --pickles ${pickles[@]}
python /home/bkmi/flowmm/src/flowmm/common/combine_manual_preprocessed.py /home/bkmi/flowmm/data/llm_mp20_alex_abridged/val_ori.pt --pickles /home/bkmi/flowmm/data/llm_mp20_alex_abridged/val.pkl
