#!/bin/bash

ALGORITHM="ddrm"
BASE_OUTDIR="images"
DATASETS="imagenet celebahq"
SCALES="8 4 2" 
mask="half" # unused, but include this for consistency with other scripts

# CoPaint
for dataset in $DATASETS
do
    for scale in $SCALES
    do
        COMMON="--dataset_name ${dataset} --n_samples 1 --config_file configs/${dataset}sp.yaml"
        OUT_PATH=${BASE_OUTDIR}/${ALGORITHM}/${dataset}/superres-${scale}/
        python main.py $COMMON --outdir $OUT_PATH --mask_type $mask --algorithm $ALGORITHM --scale ${scale}
        # have to run ddrm 2 times to generate 2 samples
        python main.py $COMMON --outdir $OUT_PATH --mask_type $mask --algorithm $ALGORITHM --seed 43 --scale $scale
    done
done
