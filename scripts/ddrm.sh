#!/bin/bash

ALGORITHM="ddrm"
BASE_OUTDIR="images"
DATASETS="imagenet celebahq"
MASKTYPES="half line sr2 expand text narrow wide"

# CoPaint
for dataset in $DATASETS
do
    for mask in $MASKTYPES
    do
        COMMON="--dataset_name ${dataset} --n_samples 1 --config_file configs/${dataset}.yaml"
        OUT_PATH=${BASE_OUTDIR}/${ALGORITHM}/${dataset}/${mask}/
        python main.py $COMMON --outdir $OUT_PATH --mask_type $mask --algorithm $ALGORITHM
        # have to run ddrm 2 times to generate 2 samples
        python main.py $COMMON --outdir $OUT_PATH --mask_type $mask --algorithm $ALGORITHM --seed 43
    done
done
