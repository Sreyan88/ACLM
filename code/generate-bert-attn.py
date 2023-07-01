import torch
from transformers import AutoModel,AutoTokenizer
import numpy as np
import re
from tqdm import tqdm
import sys
import random
import argparse
from stopwordsiso import stopwords
import pandas as pd

import os

# os.environ["CUDA_VISIBLE_DEVICES"]=""

stopwords_all = stopwords(["bn","de", "es","en","hi","ko","nl","ru","tr","zh"])

# print(stopwords_all)

parser = argparse.ArgumentParser()
parser.add_argument('-a', type=str, required=True, help="mode attn/random")
parser.add_argument('-m', type=float, required=True, help="masking rate")
parser.add_argument('-dir', type=str, required=True, help="directory to save everything")
parser.add_argument('-ckpt', type=str, required=True, help="model file path")
parser.add_argument('-tf', type=str, required=True, help="training file path")
parser.add_argument('-df', type=str, required=True, help="dev file path")


args = parser.parse_args()


mode = args.a
k = args.m
dir = args.dir

model_string = 'xlm-roberta-large'
checkpoint = args.ckpt
model = AutoModel.from_pretrained(model_string,output_hidden_states=True, output_attentions=True)
tokenizer = AutoTokenizer.from_pretrained(model_string, add_prefix_space=True)
bert_keys = list(model.state_dict().keys())
print(len(bert_keys))

ckpt = torch.load(checkpoint)

try:
    ckpt_keys = list(ckpt['state_dict'].keys())

    print(len(ckpt_keys))
    count = 0

    for i in range(len(bert_keys)):

        if bert_keys[i] in ckpt_keys[i]:

            ckpt['state_dict'][bert_keys[i]] = ckpt['state_dict'][ckpt_keys[i]]


            count += 1

    if mode=='attn':
        model.load_state_dict(ckpt['state_dict'], strict = False)
except:
    ckpt_keys = list(ckpt.keys())

    print(len(ckpt_keys))
    count = 0

    for i in range(len(bert_keys)):

        if bert_keys[i] in ckpt_keys[i]:

            ckpt[bert_keys[i]] = ckpt[ckpt_keys[i]]


            count += 1

    if mode=='attn':
        model.load_state_dict(ckpt, strict = False)



# with open("testing.txt","w") as f:
#     for i in range(len(bert_keys)):

#         f.write(str(i))
#         f.write(bert_keys[i]+"\n")
#         f.write(str(ckpt[bert_keys[i]])+"\n")
#         f.write(str(model.state_dict()[bert_keys[i]])+"\n")

print(len(model.state_dict().keys()))

print(count)





def getAttentionMatrix(sent):
    tokens = []

    sentence_split = sent.split(" ")

    x = tokenizer(sentence_split, return_tensors='pt',is_split_into_words=True)

    # print(x)

    # hggh

    word_ids = x.word_ids()



    # reverse = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(sent)

    # print(reverse)
    # print(tokenizer.convert_ids_to_tokens(x["input_ids"][0]))
    # print(word_ids)


    toDelete = []



    for i in range(len(word_ids)):
        id = word_ids[i]
        if id==None:
            toDelete = toDelete + [i]
            continue

        id = id-1

        if len(tokens)==id:
            tokens = tokens + [[i]]
        else:
            tokens[id].append(i)



    output = model(**x)


    attention_matrix = np.zeros([len(tokens), len(tokens)])

    for i in range(len(tokens)):
            toDelete = toDelete + tokens[i][1:]

    for layer in range(8,12):

        attention = output[3][layer][0].detach().numpy()

        attention = np.sum(attention, axis= 0)

        for i in range(len(tokens)):

            if(len(tokens[i])>1):
                attention[:,tokens[i][0]] = np.sum(attention[:, tokens[i][0]:tokens[i][0]+len(tokens[i])], axis=1)



        for i in range(len(tokens)):

            if(len(tokens[i])>1):
                attention[tokens[i][0],:] = np.mean(attention[tokens[i][0]:tokens[i][0]+len(tokens[i]),:], axis =0)

        attention = np.delete(attention,toDelete, axis=1)
        attention = np.delete(attention,toDelete, axis=0)


        attention_matrix = np.add(attention_matrix, attention)


    return attention_matrix


# [('colipa', 'B-LOC'), ('hypertrophy', 'O')],
# [('is', 'O'), ('the', 'O'), ('logic', 'B-CW'), ('studio', 'I-CW'), ('real', 'O')]]

# getAttentionMatrix(sequence)


def getMasksTry(sent,k):
    try:
        x,y = getMasks(sent,k)
        return x,y
    except:

        ignore = []
        ners = []
        for i in range(len(sent)):
            t = sent[i]

            if t[1].startswith('B'):
                ners.append([i])
            elif t[1].startswith('I'):
                ignore.append(i)
                ners[-1].append(i)
            elif re.search(r'\W_', t[0]) or t[0] in stopwords_all:
                # print(t[0])
                ignore.append(i)

        unmasked = []




        new_input = []
        for t in sent:
            new_input.append([t[0],t[1],0])




        for i in range(len(ignore)):
            new_input[ignore[i]][2] =0

        for i in range(len(ners)):
            for t in ners[i]:
                new_input[t][2] =1

        return new_input, unmasked




def getMasks(sent, k=0.2):

    countners=0

    sequence = ''
    ignore = []
    ners = []
    for i in range(len(sent)):
        t = sent[i]

        sequence = sequence+" "+t[0]

        if t[1].startswith('B'):
            ners.append([i])
            countners=countners+1
        elif t[1].startswith('I'):
            ignore.append(i)
            ners[-1].append(i)
            countners=countners+1
        elif re.search(r'[\u0021-\u002F\u003A-\u0040\u005B-\u0060\u007B-\u007E]', t[0])  or t[0] in stopwords_all:
            ignore.append(i)
            # print(t[0])


    k = int(np.ceil(k*(len(sent)-countners)))
    k = min(k, len(sent))

    attentionMatrix = getAttentionMatrix(sequence)



    for i in range(len(ners)):
        attentionMatrix[:,ners[i][0]] = np.sum(attentionMatrix[:, ners[i][0]:ners[i][0]+len(ners[i])], axis=1)



    for i in range(len(ners)):
        attentionMatrix[ners[i][0],:] = np.mean(attentionMatrix[ners[i][0]:ners[i][0]+len(ners[i]),:], axis =0)

    unmasked = []

    for i in range(len(attentionMatrix)):
        attentionMatrix[i][i] = -100

    attentionMatrix[:,ignore]= -100

    # print(attentionMatrix)



    # print(unmasked)

    modelInput = "[MASK]"

    new_input = []



    # try:

    for i in range(len(ners)):
        for t in ners[i]:
            unmasked.append(t)

        topK = np.argpartition(attentionMatrix[ners[i][0]], -k)[-k:]

        for t in topK:
            unmasked.append(t)

    unmasked = list(set(unmasked))

    unmasked.sort()

    # print(unmasked)



    for t in sent:
        new_input.append([t[0],t[1],0])

    # print(new_input)

    for i in range(len(unmasked)):
        new_input[unmasked[i]][2] =1



    for i in range(len(ignore)):
        new_input[ignore[i]][2] =0

    for i in range(len(ners)):
        for t in ners[i]:
            new_input[t][2] =1

        # if i==0 or unmasked[i-1] +1 != unmasked[i]:
        #     modelInput = modelInput + " "+ "[MASK]"
        # modelInput = modelInput + sent[i][0]




    # except:

    #     unmasked = []




    #     new_input = []
    #     for t in sent:
    #         new_input.append([t[0],t[1],0])

    #     for i in range(len(unmasked)):
    #         new_input[unmasked[i]][2] =1



    #     for i in range(len(ignore)):
    #         new_input[ignore[i]][2] =0

    #     for i in range(len(ners)):
    #         for t in ners[i]:
    #             new_input[t][2] =1


    #     unmasked.append("err")


    # print(new_input)
    # print(unmasked)



    return new_input, unmasked

def getMasks2(sent, k=0.2):

    countners=0

    sequence = ''
    ignore = []
    ners = []
    indexes_to_consider = []
    for i in range(len(sent)):
        t = sent[i]

        sequence = sequence+" "+t[0]

        if t[1].startswith('B'):
            ners.append([i])
            countners=countners+1
        elif t[1].startswith('I'):
            ignore.append(i)
            ners[-1].append(i)
            countners=countners+1
        elif re.search(r'[\u0021-\u002F\u003A-\u0040\u005B-\u0060\u007B-\u007E]', t[0]):
            ignore.append(i)
        else:
            indexes_to_consider.append(i)
            # print(t[0])

    k = int(np.ceil(k*len(indexes_to_consider)))
    k = min(k, len(sent))






    unmasked = random.sample(indexes_to_consider,k)



    new_input = []




    for t in sent:
        new_input.append([t[0],t[1],0])

    # print(new_input)

    for i in range(len(unmasked)):
        new_input[unmasked[i]][2] =1

    for i in range(len(ignore)):
        new_input[ignore[i]][2] =0

    for i in range(len(ners)):
        for t in ners[i]:
            new_input[t][2] =1

    return new_input, unmasked

def split_at_punctuation(s):
    delimiters = re.findall(r',', s)
    split_list = re.split(r',', s)
    result = []
    for i, item in enumerate(split_list):
        result.append(item)
        if i < len(split_list) - 1:
            result.append(delimiters[i])

    return_result = [string for string in result if (len(string) > 0 and string!='̇')]
    return return_result

def process_file(file_path):
  # Open the file
  with open(file_path) as file:
    # Create an empty list to store the results
    result = []

    # Create an empty list to store the current data
    current_list = []

    # Read each line of the file
    for line in file:
      # Remove the newline character from the line
      line = line.strip('\n')

      # Split the line by the tab character and store it in a tuple
      line_tuple = tuple(line.split('\t'))


      # Check if the tuple contains an empty string
      if not '' in line_tuple:
        # Append the tuple to the current list
        current_list.append(line_tuple)



      # If the line is blank, append the current list to the result list
      # if it's not empty, and start a new list
      if not line.strip():
        if current_list:
          result.append(current_list)
          current_list = []

    # Append the current list to the result list, in case there was no
    # blank line at the end of the file
    if current_list:
      result.append(current_list)

    # Return the result list
    return result



def process_file2(file_path):
  # Open the file
  with open(file_path) as file:
    # Create an empty list to store the results
    result = []

    # Create an empty list to store the current data
    current_list = []

    # Read each line of the file
    for line in file:
        # print(line)
      # Remove the newline character from the line
        line = line.strip('\n')

        if not line.strip():
            if current_list:
                result.append(current_list)
                current_list = []
                continue

        thisLine = line.split('\t')

        # line_tuple = tuple(line.split('\t'))

        words = split_at_punctuation(thisLine[0])

      # Split the line by the tab character and store it in a tuple
    #   print(words)

        for i in range(len(words)):
            word = words[i]

            if i==0:
                line_tuple = tuple([word,thisLine[1]])
            elif thisLine[1]=='O':
                line_tuple = tuple([word,thisLine[1]])
            else:
                line_tuple = tuple([word, str('I'+thisLine[1][1:])])


            # Check if the tuple contains an empty string
            if not '' in line_tuple:
                # Append the tuple to the current list
                current_list.append(line_tuple)



      # If the line is blank, append the current list to the result list
      # if it's not empty, and start a new list


    # Append the current list to the result list, in case there was no
    # blank line at the end of the file
    if current_list:
        result.append(current_list)

    # print(result)

    # Return the result list
    return result


def generateMasks(path, out,k, mode):
    sentences = process_file2(path)
    newsents = []
    unmasked = []



    for sent in tqdm(sentences):
        # print(sent)


        if mode=="attn" or mode=="plm" :
            newsent, unmasks= getMasksTry(sent,k)
        elif mode=="random":
            newsent,unmasks = getMasks2(sent,k)
        else:
            raise Exception("Not valid mode")







        newsents.append(newsent)
        unmasked.append(unmasks)




    with open(out,"w") as f:
        for s in newsents:
            for token in s:
                f.write(token[0]+'\t'+token[1]+'\t'+str(token[2])+"\n")
            f.write("\n")


    return newsents,unmasked




if dir[-1] != '/':
    dir += '/'


print(mode,k,dir)

train_file = args.tf.split('/')[-1].split('.')[0]
dev_file = args.df.split('/')[-1].split('.')[0]

train_file_name = dir+train_file+"_"+mode+"_"+str(k)+"_"+model_string
dev_file_name = dir+dev_file+"_"+mode+"_"+str(k)+"_"+model_string

a,b= generateMasks(args.tf, train_file_name+".txt",k, mode)
a,b= generateMasks(args.df, dev_file_name+".txt",k, mode)


def sketch9 (data):
    '''
       Saving normal text, corresponding label and bert att
    '''
    text = ''
    label_sent = ''
    bert_sent = ''
    final = []
    for i in tqdm(data):
        if i == '':
            if text!='':
                final.append([text.strip(),label_sent.strip(), bert_sent.strip()])
            text = ''
            label_sent = ''
            bert_sent = ''
            continue

        word, label, bert_att = i.split('\t')

        bert_sent += ' ' + bert_att
        text += ' ' + word
        label_sent += ' ' + label

    dataset = pd.DataFrame(final, columns=['text', 'label', 'bert_att'])
    return dataset

with open(train_file_name+".txt", 'r') as f:
    data = f.read().splitlines()

dataset = sketch9(data)
dataset.to_csv(train_file_name+".csv", index=False)

with open(dev_file_name+".txt", 'r') as f:
    data = f.read().splitlines()

dataset = sketch9(data)
dataset.to_csv(dev_file_name+".csv", index=False)