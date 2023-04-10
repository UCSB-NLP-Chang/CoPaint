#!/bin/bash

ALGORITHM="copaint-tt"
BASE_OUTDIR="images"
DATASETS="imagenet celebahq"
SCALES="8 4 2" 
mask="half" # unused, but include this for consistency with other scripts

# CoPaint
for dataset in $DATASETS
do
    for scale in $SCALES
    do
        COMMON="--dataset_name ${dataset} --n_samples 2 --config_file configs/${dataset}sp.yaml"
        OUT_PATH=${BASE_OUTDIR}/${ALGORITHM}/${dataset}/superres-${scale}/
        python main.py $COMMON  --outdir $OUT_PATH \
                                --mask_type $mask \
                                --scale ${scale} \
                                --algorithm o_ddim \
                                --ddim.schedule_params.jump_length 10 \
                                --ddim.schedule_params.jump_n_sample 2 \
                                --ddim.schedule_params.use_timetravel
    done
done
