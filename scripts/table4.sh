
#!/bin/bash

TAG=table_4

DEVICE=0
############################### GENERATE SCORE FILES ###########################
# for condense_key in 'idc' 'dream' 'mtt' 'dsa'; do
# for condense_key in 'dsa'; do
#     CUDA_VISIBLE_DEVICES=$DEVICE python get_training_dynamics.py --dataset cifar10 --ipc 10 --condense_key $condense_key --run_tag ${TAG}
#     CUDA_VISIBLE_DEVICES=$DEVICE python generate_importance_score.py --dataset cifar10 --ipc 10 --condense_key $condense_key
#     wait
# done

############################### EVALUATE ###########################
EXTRA_PARA="--no_resnet --repeat 3 --run_tag ${TAG}"
for condense_key in 'idc' 'dream' 'mtt' 'dsa'; do
    CUDA_VISIBLE_DEVICES=$DEVICE python test.py -d cifar10 $EXTRA_PARA --ipc 10 --slct_ipc 1 --pruning_key accumulated_margin --condense_key $condense_key
    CUDA_VISIBLE_DEVICES=$DEVICE python test.py -d cifar10 $EXTRA_PARA --ipc 10 --slct_ipc 1 --pruning_key accumulated_margin_easy --condense_key $condense_key
    CUDA_VISIBLE_DEVICES=$DEVICE python test.py -d cifar10 $EXTRA_PARA --ipc 10 --slct_ipc 1 --pruning_key forgetting --condense_key $condense_key
    CUDA_VISIBLE_DEVICES=$DEVICE python test.py -d cifar10 $EXTRA_PARA --ipc 10 --slct_ipc 1 --pruning_key forgetting_easy --condense_key $condense_key
    CUDA_VISIBLE_DEVICES=$DEVICE python test.py -d cifar10 $EXTRA_PARA --ipc 10 --slct_ipc 1 --pruning_key el2n --condense_key $condense_key
    CUDA_VISIBLE_DEVICES=$DEVICE python test.py -d cifar10 $EXTRA_PARA --ipc 10 --slct_ipc 1 --pruning_key el2n_easy --condense_key $condense_key
    CUDA_VISIBLE_DEVICES=$DEVICE python test.py -d cifar10 $EXTRA_PARA --ipc 10 --slct_ipc 1 --pruning_key yoco --condense_key $condense_key
    wait
done