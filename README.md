# ACLM

Implementation of [ACLM: A Selective-Denoising based Generative Data Augmentation Approach for Low-Resource Complex NER](https://arxiv.org/abs/2306.00928)

![Proposed Methodology](./assets/diagram.jpg)

* Installing dependencies using:
```
pip install -r requirements.txt
```

Steps:

1. Use [generate-bert-attn.py](./generate-bert-attn.py) to process the files in [data](./data/) \\
2. Update the paths and required file names in [train_dynamic_multilingual.sh](./train_dynamic_multilingual.sh) and [train_dynamic_multilingual_mixup.sh](./train_dynamic_multilingual_mixup.sh)
3. Run the sh files

train_dynamic_multilingual.sh - ACLM
train_dynamic_multilingual_mixup.sh - ACLM with mixner