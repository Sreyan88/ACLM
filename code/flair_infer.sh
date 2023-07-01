#!/bin/bash

#SBATCH -t 23:59:00
#SBATCH --nodes 1
#SBATCH -n 1
#SBATCH -c 10
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=a100:1
#SBATCH --partition=gpu
#SBATCH --output=./logs/cross/infer_cross_200_54.99.out
#SBATCH --error=./logs/cross/infer_cross_200_54.99.out

# 100,200,8
# 500,1000,16
# array=( bn de en fa hi ko multi nl ru tr zh )

set -e
set -x

# 500 - /home/sreyang/scratch/utkarsh/Complex-NER/low_res_samples/iterative_sampling/500/en_sample_train_attn_0.3_xlm-roberta-large-0.3-false-none-attention-dynamic-0.3-5-false-500-xlm-large-en-mixup-42-new-mixup-flair/best-model.pt
# 100 - /home/sreyang/scratch/utkarsh/Complex-NER/low_res_samples/iterative_sampling/100/en_sample_train_attn_0.3_xlm-roberta-large-no-random-0.3-false-gauss-attention-dynamic-0.3-5-false-100-xlm-large-en-mixup-40-mixup-flair/best-model.pt
# 200 - /home/sreyang/scratch/utkarsh/Complex-NER/low_res_samples/iterative_sampling/200/en_sample_train_attn_0.3_xlm-roberta-large-0.3-false-all-attention-dynamic-0.3-5-false-200-xlm-large-en-mixup-42-mixup-flair/best-model.pt
# 1000 -  /home/sreyang/scratch/utkarsh/Complex-NER/low_res_samples/iterative_sampling/1000/en_sample_train_attn_0.3_xlm-roberta-large-0.3-false-gauss-attention-dynamic-0.3-5-false-1000-xlm-large-en-mixup-42-retrain-mixup-flair/best-model.pt

array1=( hi bn de zh )
array2=( 200 )

for i in "${array1[@]}"
do
    for j in "${array2[@]}"
    do
        directory=/home/sreyang/scratch/utkarsh/Complex-NER/low_res_samples/iterative_sampling/${j}/
        dev_file=/home/sreyang/scratch/utkarsh/Complex-NER/low_res_samples/iterative_sampling/${j}/${i}_dev.conll
        test_file=/home/sreyang/scratch/utkarsh/Complex-NER/low_res_samples/iterative_sampling/${j}/${i}_test.conll
        # ckpt=/home/sreyang/scratch/acl_2/new/Complex-NER/dynamic-bart-pretraining-script/logs/mr/flair_${j}_en/best-model.pt
        ckpt=/home/sreyang/scratch/acl_2/new/Complex-NER/multilingual/utkarsh_code/data/500/en_sample_train_attn_0.3_xlm-roberta-large-no-random-0.3-false-all-attention-dynamic-0.3-false-500-xlm-large-en-42-flair-2/best-model.pt

        # output=/home/sreyang/scratch/acl_2/new/Complex-NER/dynamic-bart-pretraining-script/logs/mr/flair_${j}_en/
        output=/home/sreyang/scratch/utkarsh/Complex-NER/low_res_samples/iterative_sampling/${j}/cross/flair_${j}_en/

        python flair_infer.py \
        --input_folder $directory \
        --output_folder $output \
        --dev_file $dev_file \
        --gpu cuda:0 \
        --input_file $test_file \
        --checkpoint $ckpt
    done

done


# directory=/home/sreyang/scratch/acl_2/new/Complex-NER/dynamic-bart-pretraining-script/logs/data

# train_file=/home/sreyang/scratch/acl_2/push/Complex-NER/low_res_samples/iterative_sampling/100/mr/en_sample_train_lwtr_new.conll
# dev_file=/home/sreyang/scratch/acl_2/push/Complex-NER/low_res_samples/iterative_sampling/100/en_dev.conll
# test_file=/home/sreyang/scratch/acl_2/push/Complex-NER/low_res_samples/iterative_sampling/100/en_test.conll

# python flair_train.py \
# --input_folder $directory \
# --output_folder "logs/mr/flair_100_en_2" \
# --gpu cuda:0 \
# --train_file $train_file \
# --dev_file $dev_file \
# --test_file  $test_file \
# --batch_size 8 \
# --lr 0.01 \
# --epochs 100 \
# --seed 42