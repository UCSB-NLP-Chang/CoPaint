#!/bin/bash

ALGORITHM="copaint-fast"
BASE_OUTDIR="images"
DATASETS="imagenet celebahq"
MASKTYPES="half line sr2 expand text narrow wide"

# CoPaint
for dataset in $DATASETS
do
    for mask in $MASKTYPES
    do
        COMMON="--dataset_name ${dataset} --n_samples 2 --config_file configs/${dataset}.yaml"
        OUT_PATH=${BASE_OUTDIR}/${ALGORITHM}/${dataset}/${mask}/

        # if dataset equals to imagenet then 
        if [ $dataset = "imagenet" ]; then
            LR=0.025
        # else dataset equals to celebahq then 
        else
            LR=0.02
        fi

        python main.py $COMMON --outdir $OUT_PATH \
                               --mask_type $mask \
                               --algorithm o_ddim \
                               --ddim.schedule_params.num_inference_steps 100 \
                               --optimize_xt.num_iteration_optimize_xt 1 \
                               --optimize_xt.lr_xt $LR \
                               --optimize_xt.lr_xt_decay 1.03
    done
done
