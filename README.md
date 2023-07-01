# ACLM

Implementation of [ACLM: A Selective-Denoising based Generative Data Augmentation Approach for Low-Resource Complex NER](https://arxiv.org/abs/2306.00928)

![Proposed Methodology](./assets/diagram.jpg)

* Installing dependencies using:
```
pip install -r requirements.txt
```

Steps:

1. Use [generate-bert-attn.py](./src/generate-bert-attn.py) to process the files in [data](./data/) \
2. cd ./src
3. Update the paths and required file names in [train_dynamic_multilingual.sh](./src/train_dynamic_multilingual.sh) and [train_dynamic_multilingual_mixup.sh](./src/train_dynamic_multilingual_mixup.sh)
4. Run the sh files

train_dynamic_multilingual.sh - ACLM \
train_dynamic_multilingual_mixup.sh - ACLM with mixner


Languages Used and their keys:
German (de_DE), English (en_XX), Spanish (es_XX), Hindi (hi_IN), Korean (ko_KR), Dutch (nl_XX), Russian (ru_RU), Turkish (tr_TR), Chinese (zh_CN), Bengali (bn_IN)