#!/bin/bash

TAG=table_1
DEVICE=1

######################################### Table 1: GENERATE SCORE #########################################
# Use downloaded dynamics
# Usage: CUDA_VISIBLE_DEVICES=<DEVICE> <FUNCTION> <DATASET> <IPC>
# for DATASET in "cifar10" "cifar100" "imagenet"; do
#     if [ "$DATASET" = 'cifar10' ]; then
#         IPCF_LIST=(10 50)
#     elif [ "$DATASET" = 'cifar100' ]; then
#         IPCF_LIST=(10 20 50)
#     elif [ "$DATASET" = 'imagenet' ]; then
#         IPCF_LIST=(10 20)
#     fi
#     for IPC in ${IPCF_LIST[@]}; do
#         CUDA_VISIBLE_DEVICES=$DEVICE python get_training_dynamics.py --dataset $DATASET --ipc $IPC --run_tag $TAG
#         CUDA_VISIBLE_DEVICES=$DEVICE python generate_importance_score.py --dataset $DATASET --ipc $IPC
#         echo
#         wait
#     done
# done

######################################### Table 1: EVALUATE SCORE #########################################
EXTRA_PARA="--no_resnet --repeat 1 --run_tag ${TAG}"

for DATASET in "cifar10" "cifar100" "imagenet"; do
    if [ "$DATASET" = 'cifar10' ]; then
        IPCF_LIST=(10)
    elif [ "$DATASET" = 'cifar100' ]; then
        IPCF_LIST=(10)
    elif [ "$DATASET" = 'imagenet' ]; then
        IPCF_LIST=(10)
    fi
    for IPC in ${IPCF_LIST[@]}; do
        IPCT_LIST=(1)
        if [ "$IPC" = "50" ]; then
            IPCT_LIST=(1)
        fi
        for IPCT in ${IPCT_LIST[@]}; do
            CUDA_VISIBLE_DEVICES=$DEVICE python test.py -d $DATASET $EXTRA_PARA --ipc $IPC --slct_ipc $IPCT --pruning_key random &
            if [ "$DATASET" != 'imagenet' ]; then
                CUDA_VISIBLE_DEVICES=$DEVICE python test.py -d $DATASET $EXTRA_PARA --ipc $IPC --slct_ipc $IPCT --pruning_key random --condense_key dream &
            fi
            CUDA_VISIBLE_DEVICES=$DEVICE python test.py -d $DATASET $EXTRA_PARA --ipc $IPC --slct_ipc $IPCT --pruning_key ssp &
            CUDA_VISIBLE_DEVICES=$DEVICE python test.py -d $DATASET $EXTRA_PARA --ipc $IPC --slct_ipc $IPCT --pruning_key entropy &
            CUDA_VISIBLE_DEVICES=$DEVICE python test.py -d $DATASET $EXTRA_PARA --ipc $IPC --slct_ipc $IPCT --pruning_key accumulated_margin &
            CUDA_VISIBLE_DEVICES=$DEVICE python test.py -d $DATASET $EXTRA_PARA --ipc $IPC --slct_ipc $IPCT --pruning_key forgetting &
            CUDA_VISIBLE_DEVICES=$DEVICE python test.py -d $DATASET $EXTRA_PARA --ipc $IPC --slct_ipc $IPCT --pruning_key el2n &
            CUDA_VISIBLE_DEVICES=$DEVICE python test.py -d $DATASET $EXTRA_PARA --ipc $IPC --slct_ipc $IPCT --pruning_key ccs &
            CUDA_VISIBLE_DEVICES=$DEVICE python test.py -d $DATASET $EXTRA_PARA --ipc $IPC --slct_ipc $IPCT --pruning_key yoco &
            wait
        done
        wait
    done
    wait
done

# EXTRA_PARA="--no_resnet --repeat 3 --run_tag ${TAG}"

# for DATASET in "cifar10" "cifar100" "imagenet"; do
#     if [ "$DATASET" = 'cifar10' ]; then
#         IPCF_LIST=(10 50)
#     elif [ "$DATASET" = 'cifar100' ]; then
#         IPCF_LIST=(10 20 50)
#     elif [ "$DATASET" = 'imagenet' ]; then
#         IPCF_LIST=(10 20)
#     fi
#     for IPC in ${IPCF_LIST[@]}; do
#         IPCT_LIST=(1 2 5)
#         if [ "$IPC" = "50" ]; then
#             IPCT_LIST=(1 2 5 10)
#         fi
#         for IPCT in ${IPCT_LIST[@]}; do
#             CUDA_VISIBLE_DEVICES=$DEVICE python test.py -d $DATASET $EXTRA_PARA --ipc $IPC --slct_ipc $IPCT --pruning_key random
#             if [ "$DATASET" != 'imagenet' ]; then
#                 CUDA_VISIBLE_DEVICES=$DEVICE python test.py -d $DATASET $EXTRA_PARA --ipc $IPC --slct_ipc $IPCT --pruning_key random --condense_key dream
#             fi
#             CUDA_VISIBLE_DEVICES=$DEVICE python test.py -d $DATASET $EXTRA_PARA --ipc $IPC --slct_ipc $IPCT --pruning_key ssp
#             CUDA_VISIBLE_DEVICES=$DEVICE python test.py -d $DATASET $EXTRA_PARA --ipc $IPC --slct_ipc $IPCT --pruning_key entropy
#             CUDA_VISIBLE_DEVICES=$DEVICE python test.py -d $DATASET $EXTRA_PARA --ipc $IPC --slct_ipc $IPCT --pruning_key accumulated_margin
#             CUDA_VISIBLE_DEVICES=$DEVICE python test.py -d $DATASET $EXTRA_PARA --ipc $IPC --slct_ipc $IPCT --pruning_key forgetting
#             CUDA_VISIBLE_DEVICES=$DEVICE python test.py -d $DATASET $EXTRA_PARA --ipc $IPC --slct_ipc $IPCT --pruning_key el2n
#             CUDA_VISIBLE_DEVICES=$DEVICE python test.py -d $DATASET $EXTRA_PARA --ipc $IPC --slct_ipc $IPCT --pruning_key ccs
#             CUDA_VISIBLE_DEVICES=$DEVICE python test.py -d $DATASET $EXTRA_PARA --ipc $IPC --slct_ipc $IPCT --pruning_key yoco
#             wait
#         done
#         wait
#     done
#     wait
# done