#!/bin/bash

set -e
set -x

array1=( hi bn de zh )
array2=( 200 )

for i in "${array1[@]}"
do
    for j in "${array2[@]}"
    do
        directory=../data/${j}/
        dev_file=../data/${j}/${i}_dev.conll
        test_file=../data/${j}/${i}_test.conll
        ckpt=/home/sreyang/scratch/acl_2/new/Complex-NER/multilingual/utkarsh_code/data/500/en_sample_train_attn_0.3_xlm-roberta-large-no-random-0.3-false-all-attention-dynamic-0.3-false-500-xlm-large-en-42-flair-2/best-model.pt
        output=../data/${j}/cross/flair_${j}_en/

        python flair_infer.py \
        --input_folder $directory \
        --output_folder $output \
        --dev_file $dev_file \
        --gpu cuda:0 \
        --input_file $test_file \
        --checkpoint $ckpt
    done

done