#!/bin/bash

(
# download pretrained models
mkdir -p checkpoints
cd checkpoints

# model pretrained on ImageNet
wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_classifier.pt # Trained by OpenAI
wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion.pt  # Trained by OpenAI
wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/512x512_classifier.pt
wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/512x512_diffusion.pt

# model pretrained on CelebA
gdown https://drive.google.com/uc?id=1norNWWGYP3EZ_o05DmoW1ryKuKMmhlCX
)

prepare_celeba() {
    # Copied from [Lama](https://github.com/saic-mdal/lama)
    BASENAME="lama-celeba"
    mkdir -p $BASENAME

    unzip data256x256.zip -d ${BASENAME}

    # Reindex
    for i in `echo {00001..30000}`
    do
        mv ${BASENAME}'/data256x256/'$i'.jpg' ${BASENAME}'/data256x256/'$[10#$i - 1]'.jpg'
    done

    # Split: split train -> train & val
    cat lama_split/train_shuffled.flist | shuf > ${BASENAME}/temp_train_shuffled.flist
    cat ${BASENAME}/temp_train_shuffled.flist | head -n 2000 > ${BASENAME}/val_shuffled.flist
    cat ${BASENAME}/temp_train_shuffled.flist | tail -n +2001 > ${BASENAME}/train_shuffled.flist
    cat lama_split/val_shuffled.flist > ${BASENAME}/visual_test_shuffled.flist

    mkdir ${BASENAME}/train_256/
    mkdir ${BASENAME}/val_source_256/
    mkdir ${BASENAME}/visual_test_source_256/

    cat ${BASENAME}/train_shuffled.flist | xargs -I {} mv ${BASENAME}/data256x256/{} ${BASENAME}/train_256/
    cat ${BASENAME}/val_shuffled.flist | xargs -I {} mv ${BASENAME}/data256x256/{} ${BASENAME}/val_source_256/
    cat ${BASENAME}/visual_test_shuffled.flist | xargs -I {} mv ${BASENAME}/data256x256/{} ${BASENAME}/visual_test_source_256/
}

# download datasets 
(
mkdir -p datasets
cd datasets
# cd datasets
# celeba data
gdown https://drive.google.com/uc?id=1O89DVCoWsMhrIF3G8-wMOJ0h7LukmMdP
sleep 1
prepare_celeba
rm data256x256.zip
# imagenet data
# you can find it in datasets/imagenet100
)