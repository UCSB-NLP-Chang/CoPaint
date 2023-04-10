#!/bin/bash

ALGORITHM="copaint-tt"
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
        python main.py $COMMON  --outdir $OUT_PATH \
                                --mask_type $mask \
                                --algorithm o_ddim \
                                --ddim.schedule_params.jump_length 10 \
                                --ddim.schedule_params.jump_n_sample 2 \
                                --ddim.schedule_params.use_timetravel
    done
done
