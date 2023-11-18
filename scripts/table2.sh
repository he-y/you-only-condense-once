#!/bin/bash

TAG=table_2

DEVICE=0

############################### GENERATE SCORE FILES ###########################
# SKIP IDC, USE PREVIOUSLY GENERATED SCORE FILES
# for condense_key in 'dream' 'mtt' 'kip'; do
#     CUDA_VISIBLE_DEVICES=$DEVICE python get_training_dynamics.py --dataset cifar10 --ipc 10 --condense_key $condense_key --run_tag $TAG
#     CUDA_VISIBLE_DEVICES=$DEVICE python generate_importance_score.py --dataset cifar10 --ipc 10 --condense_key $condense_key
#     wait
# done

############################### EVALUATE ###########################
EXTRA_PARA="--no_resnet --repeat 3 --run_tag ${TAG}"

for condense_key in 'idc' 'dream' 'mtt' 'kip'; do
    CUDA_VISIBLE_DEVICES=$DEVICE python test.py -d cifar10 $EXTRA_PARA --ipc 10 --slct_ipc 1 --pruning_key random --condense_key $condense_key &
    CUDA_VISIBLE_DEVICES=$DEVICE python test.py -d cifar10 $EXTRA_PARA --ipc 10 --slct_ipc 1 --pruning_key random_balance --condense_key $condense_key &
    CUDA_VISIBLE_DEVICES=$DEVICE python test.py -d cifar10 $EXTRA_PARA --ipc 10 --slct_ipc 1 --pruning_key yoco_imbalance --condense_key $condense_key &
    CUDA_VISIBLE_DEVICES=$DEVICE python test.py -d cifar10 $EXTRA_PARA --ipc 10 --slct_ipc 1 --pruning_key yoco --condense_key $condense_key &
    wait
done