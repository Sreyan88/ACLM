#!/bin/bash

# All languages and their keys in m-bart

# Arabic (ar_AR), Czech (cs_CZ), German (de_DE), English (en_XX), Spanish (es_XX), Estonian (et_EE),
# Finnish (fi_FI), French (fr_XX), Gujarati (gu_IN), Hindi (hi_IN), Italian (it_IT),
# Japanese (ja_XX), Kazakh (kk_KZ), Korean (ko_KR), Lithuanian (lt_LT), Latvian (lv_LV), Burmese (my_MM),
#  Nepali (ne_NP), Dutch (nl_XX), Romanian (ro_RO), Russian (ru_RU), Sinhala (si_LK),
#  Turkish (tr_TR), Vietnamese (vi_VN), Chinese (zh_CN), Afrikaans (af_ZA), Azerbaijani (az_AZ),
#  Bengali (bn_IN), Persian (fa_IR), Hebrew (he_IL), Croatian (hr_HR), Indonesian (id_ID),
#  Georgian (ka_GE), Khmer (km_KH), Macedonian (mk_MK), Malayalam (ml_IN), Mongolian (mn_MN),
#  Marathi (mr_IN), Polish (pl_PL), Pashto (ps_AF), Portuguese (pt_XX), Swedish (sv_SE),
#  Swahili (sw_KE), Tamil (ta_IN), Telugu (te_IN), Thai (th_TH), Tagalog (tl_XX), Ukrainian (uk_UA),
#  Urdu (ur_PK), Xhosa (xh_ZA), Galician (gl_ES), Slovene (sl_SI)

set -e
set -x

language=$1
language_label=$2
size=$3
flair_batch_size=$4
SEED=$5
masking_rate=$6
generations=$7

input_folder="../data/${size}"

python flair_train.py \
-i $input_folder \
-o "${input_folder}/${language}_flair_xlm_${size}" \
-g cuda:0 \
-tf "${language}_sample_train.conll" \
-bs $flair_batch_size \
-l 0.01 \
-ep 100 \
-lang $language \
-s $SEED

python generate-bert-attn.py \
-a attn \
-m $masking_rate \
-dir "../data/${size}" \
-ckpt "../data/${size}/${language}_flair_xlm_${size}/best-model.pt" \
-tf "../data/${size}/${language}_sample_train.conll" \
-df "../data/${size}/${language}_dev.conll"

directory="../data/${size}"
attn_train="${language}_sample_train_attn_${masking_rate}_xlm-roberta-large"
attn_dev="${language}_dev_attn_${masking_rate}_xlm-roberta-large"

run="${masking_rate}-false-gauss-attention-dynamic-${masking_rate}-${generations}-false-${size}-xlm-large-${language}-mixup-${SEED}-retrain"

python bart_pretrain_dynamic_multilingual.py \
--directory $directory \
--train_file $attn_train \
--dev_file $attn_dev \
--epochs 10 \
--batch_size 16 \
--mask_entities False \
--mask_attn gauss \
--mode attn \
--file_name $run \
--lang $language_label \
--seed $SEED

best_model="${directory}/${attn_train}-${run}-final"

inference_file="${language}_sample_train_attn_${masking_rate}_xlm-roberta-large"

python test-dynamic_multilingual-mixup.py \
--model $best_model \
--input_file $inference_file \
--sample_generation_mode dynamic \
--directory $directory \
--mask_entities False \
--mask_attn gauss \
--mode attn \
--topk 10 \
--num_of_sequences $generations \
--max_length 100 \
--do_sample True \
--num_beams 5 \
--file_name $run \
--root_dir $directory \
--lang $language_label \
--remove_repetitions False \
--seed $SEED

generated_file="${inference_file}-${run}-mixup"

python flair_eval_equal.py \
--input_folder $directory \
--output_folder "${directory}/${generated_file}" \
--gpu cuda:0 \
--input_file $generated_file \
--need_consistency True \
--file_name $run \
-gfl $language \
--seed $SEED \
--ckpt "../data/${size}/${language}_flair_xlm_${size}/best-model.pt"

consistent_file="${generated_file}-aug+gold.txt"

python flair_train.py \
--input_folder $directory \
--output_folder "${directory}/${generated_file}-flair" \
--gpu cuda:0 \
--train_file $consistent_file \
--batch_size $flair_batch_size \
--lr 0.01 \
--epochs 100 \
--language $language \
--seed $SEED