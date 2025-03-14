#!/bin/bash
pickles=(/fsx-open-catalyst/bkmi/flowmm/data/llm_osc/train*.pkl)
python /fsx-open-catalyst/bkmi/flowmm/src/flowmm/common/combine_manual_preprocessed.py /fsx-open-catalyst/bkmi/flowmm/data/llm_osc/train_ori.pt --pickles ${pickles[@]}
python /fsx-open-catalyst/bkmi/flowmm/src/flowmm/common/combine_manual_preprocessed.py /fsx-open-catalyst/bkmi/flowmm/data/llm_osc/val_ori.pt --pickles /fsx-open-catalyst/bkmi/flowmm/data/llm_osc/val_t1.5.pkl
