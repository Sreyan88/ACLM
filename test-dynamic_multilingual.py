import argparse
import transformers
import torch
parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument('--model', help='which model to use')
parser.add_argument('--input_file','-i', help='input file to use')
parser.add_argument('--sample_generation_mode', help='static/dynamic generation')
parser.add_argument('--directory','-dir', default='attn', help='data directory where train and dev files are located')
parser.add_argument('--mask_entities', default='False', help='Should we mask the entities following the gaussian distribution')
parser.add_argument('-a', '--mask_attn', default='all', help='Which attn mode to select (all/gauss/none)')
parser.add_argument('--mode','-m', default='attn', help='masking mode (both/either/attn)')
parser.add_argument('--topk', default='50', help='topk value')
parser.add_argument('--num_of_sequences', default=5, help='number of sequences per datapoint')
parser.add_argument('--max_length', default=100, help='max_length of the generated sentence')
parser.add_argument('--do_sample', default='True', help='do_sample argument')
parser.add_argument('--num_beams', default=5, help='num_beams argument')
parser.add_argument('--file_name','-f', type=str, default='', help='file name for output')
parser.add_argument('--root_dir','-ro', type=str, default='', help='root directory')
parser.add_argument('--lang','-la', type=str, default='', help='language you are working on')
parser.add_argument('--remove_repetitions', default='True', help="should remove repetitions?")
parser.add_argument('--seed', '-s', type=int, default=-1, help='random seed')
args = parser.parse_args()

args.remove_repetitions = False if args.remove_repetitions=='False' else True
args.mask_entities = False if args.mask_entities=='False' else True
args.do_sample = False if args.do_sample=='False' else True

print(args)
if not args.seed==-1:
    transformers.set_seed(args.seed)
    torch.backends.cudnn.deterministic = True

from transformers import pipeline
from tqdm import tqdm
from transformers import MBartForConditionalGeneration, MBart50Tokenizer
tokenizer = MBart50Tokenizer.from_pretrained(args.model)
tokenizer.src_lang = args.lang
model_pipeline = pipeline("text2text-generation", model=args.model, tokenizer=tokenizer, device=0)
import os
import pandas as pd
import random
import sys
from utils import mask_entities, mask_words,get_random_gauss_value


if args.directory[-1]!='/':
    args.directory += '/'

newFileName = args.directory + args.input_file + '.csv'

data = pd.read_csv(newFileName)
label = list(data.label.values)
text = list(data.text.values)
bert_att = list(data.bert_att.values)

new_tokens = ['<b-corp>', '<b-cw>', '<b-grp>', '<b-loc>', '<b-per>', '<b-prod>', '<i-corp>', '<i-cw>', '<i-grp>', '<i-loc>', '<i-per>', '<i-prod>']
new_tokens = set(new_tokens)

def remove_tags(temp):
    return ' '.join([i for i in temp.split() if i[0]!='<'])

def isGeneratedSentenceValid(sent):
    global new_tokens

    count = 0
    for i in sent.split(' '):
        if i!='':
            if (i[0]=='<' and i[-1]!='>') or (i[0]!='<' and i[-1]=='>'):
                return False

            if i[0]=='<' and i[-1]=='>':
                if not i in new_tokens:
                    return False
                count+=1
    if count%2:
        return False

    return True

# args.model[:-6].strip(args.file_name) + '-' +
generated_file = args.root_dir + "/" + args.input_file + '-' + args.file_name + '.txt'
if os.path.exists(generated_file):
    os.remove(generated_file)
print(generated_file)

# DYNAMIC MASKING NEW CODE
if args.sample_generation_mode=='static':
    with open(generated_file, 'w') as the_file:
        test = 0
        for i in tqdm(range(len(text))):
            saved = {}
            new_text = text[i].split()
            new_label = label[i].split()
            new_bert_attn = bert_att[i].split()
            if args.mode == 'attn':
                new_sketch = mask_entities(new_text, new_label, False)
                new_sketch = mask_words(new_sketch, new_label, new_bert_attn, args.mask_attn)
            elif args.mode == 'both':
                new_sketch = mask_entities(new_text, new_label, args.mask_entities)
                new_sketch = mask_words(new_sketch, new_label, new_bert_attn, args.mask_attn)
            elif args.mode == 'either':
                temp = get_random_gauss_value(0.5,0.3)
                if temp<=0.5:
                    new_sketch = mask_entities(new_text, new_label, False)
                    new_sketch = mask_words(new_sketch, new_label, new_bert_attn, args.mask_attn)
                else:
                    new_sketch = mask_entities(new_text, new_label, args.mask_entities)
                    new_sketch = mask_words(new_sketch, new_label, new_bert_attn, 'none')
            # the_file.write('Original: ' + text[i] + '\n')
            generated_text = model_pipeline(new_sketch, num_beams=int(args.num_beams), top_k=int(args.topk), do_sample=args.do_sample, max_length=int(args.max_length), num_return_sequences=int(args.num_of_sequences), forced_bos_token_id=tokenizer.lang_code_to_id[args.lang])
            for z in range(int(args.num_of_sequences)):
                if args.remove_repetitions:
                    if generated_text[z]['generated_text'] in saved.keys():
                        continue
                    else:
                        saved[generated_text[z]['generated_text']] = 1

                if not isGeneratedSentenceValid(generated_text[z]['generated_text']):
                    test+=1
                    continue

                # the_file.write(f'Mask {z}: ' + ' '.join(new_sketch) + '\n')
                # the_file.write(f'Generated {z}: '+ remove_tags(generated_text[z]['generated_text'])+ '\n')
                prev_label = ''
                temp = False
                for k in generated_text[z]['generated_text'].split(' '):
                    if k=='':
                        continue
                    if prev_label=='' and k[0]!='<':
                        the_file.write(f'{k}\tO\n')
                    elif prev_label!='' and k[0]=='<':
                        the_file.write(f'\t{prev_label}\n')
                        prev_label=''
                        temp = False
                        continue
                    elif k[0]=='<':
                        prev_label = k[1:-1].upper()
                        continue
                    else:
                        if temp:
                            the_file.write(f' {k}')
                        else:
                            temp = True
                            the_file.write(f'{k}')

                the_file.write('\n')
            # the_file.write('\n')
elif args.sample_generation_mode=='dynamic':
    with open(generated_file, 'w') as the_file:
        test = 0
        for i in tqdm(range(len(text))):
            saved = {}
            for z in range(int(args.num_of_sequences)):
                new_text = text[i].split()
                new_label = label[i].split()
                new_bert_attn = bert_att[i].split()
                if args.mode == 'attn':
                    new_sketch = mask_entities(new_text, new_label, False)
                    new_sketch = mask_words(new_sketch, new_label, new_bert_attn, args.mask_attn)
                elif args.mode == 'both':
                    new_sketch = mask_entities(new_text, new_label, args.mask_entities)
                    new_sketch = mask_words(new_sketch, new_label, new_bert_attn, args.mask_attn)
                elif args.mode == 'either':
                    temp = get_random_gauss_value(0.5,0.3)
                    if temp<=0.5:
                        new_sketch = mask_entities(new_text, new_label, False)
                        new_sketch = mask_words(new_sketch, new_label, new_bert_attn, args.mask_attn)
                    else:
                        new_sketch = mask_entities(new_text, new_label, args.mask_entities)
                        new_sketch = mask_words(new_sketch, new_label, new_bert_attn, 'none')

                # print(new_sketch)
                generated_text = model_pipeline(new_sketch, num_beams=int(args.num_beams), top_k=int(args.topk), do_sample=args.do_sample, max_length=int(args.max_length))
                # print(generated_text[0]['generated_text'])
                # continue
                if args.remove_repetitions:
                    if generated_text[0]['generated_text'] in saved.keys():
                        continue
                    else:
                        saved[generated_text[0]['generated_text']] = 1

                if not isGeneratedSentenceValid(generated_text[0]['generated_text']):
                    test+=1
                    continue

                # the_file.write(f'Mask {z}: ' + ' '.join(new_sketch) + '\n')
                # the_file.write(f'Generated {z}: '+ remove_tags(generated_text[z]['generated_text'])+ '\n')
                prev_label = ''
                temp = False
                for k in generated_text[0]['generated_text'].split(' '):
                    if k=='':
                        continue
                    if prev_label=='' and k[0]!='<':
                        the_file.write(f'{k}\tO\n')
                    elif prev_label!='' and k[0]=='<':
                        the_file.write(f'\t{prev_label}\n')
                        prev_label=''
                        temp = False
                        continue
                    elif k[0]=='<':
                        prev_label = k[1:-1].upper()
                        continue
                    else:
                        if temp:
                            the_file.write(f' {k}')
                        else:
                            temp = True
                            the_file.write(f'{k}')

                the_file.write('\n')
            # the_file.write('\n')
            # the_file.write('----\n\n')

print('File generated at: ', generated_file)