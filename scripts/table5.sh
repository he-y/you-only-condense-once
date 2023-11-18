#!/bin/bash

TAG=table_5

DEVICE=0

############################### GENERATE SCORE FILES ###########################
# # USE PREVIOUSLY GENERATED SCORE FILES
# condense_key=idc
# CUDA_VISIBLE_DEVICES=$DEVICE python get_training_dynamics.py --dataset cifar10 --ipc 10 --condense_key $condense_key
# CUDA_VISIBLE_DEVICES=$DEVICE python generate_importance_score.py --dataset cifar10 --ipc 10 --condense_key $condense_key

############################### EVALUATE ###########################
EXTRA_PARA="--no_resnet --repeat 3 --run_tag ${TAG}"
for IPCT in 1 2 5; do
    for PRUNING_METHOD in 'random' 'ssp' 'accumulated_margin' 'entropy' 'forgetting' 'el2n' 'yoco'; do
        CUDA_VISIBLE_DEVICES=$DEVICE python test.py -d cifar10 $EXTRA_PARA --ipc 10 --slct_ipc $IPCT --pruning_key "${PRUNING_METHOD}_imbalance"
        CUDA_VISIBLE_DEVICES=$DEVICE python test.py -d cifar10 $EXTRA_PARA --ipc 10 --slct_ipc $IPCT --pruning_key "${PRUNING_METHOD}_balance"
    done
done