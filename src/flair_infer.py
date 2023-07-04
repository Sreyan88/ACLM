import flair
import torch
import argparse
import os
from tqdm import tqdm
os.environ['CUDA_VISIBLE_DEVICES']="0"

parser = argparse.ArgumentParser(description='Train flair model')
parser.add_argument('--input_folder', '-i', help='Name of the input folder containing train, dev and test files')
parser.add_argument('--output_folder', '-o', help='Name of the input folder containing train, dev and test files')
parser.add_argument('--dev_file', '-d', help='Name of the input folder containing train, dev and test files')
parser.add_argument('--gpu', '-g', help='Use gpu/cpu, put "cuda" if gpu and "cpu" if cpu')
parser.add_argument('--checkpoint', '-ckpt', help='path to checkpoint')
parser.add_argument('--input_file', '-if', help='Name of the test file')

args = parser.parse_args()
if args.input_folder[-1]!='/':
    args.input_folder += '/'
input_folder=args.input_folder
output_folder=args.output_folder
gpu_type=args.gpu

flair.device = torch.device(gpu_type)
from typing import List
from flair.data import Sentence
from flair.models import SequenceTagger
from tqdm import tqdm
from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.trainers import ModelTrainer
from flair.models import SequenceTagger
# from flair.embeddings import *
from flair.embeddings import TransformerWordEmbeddings

# https://drive.google.com/file/d/1Evf2UjlBFP2N-5jfgpOdo2uCvh53UnX3/view?usp=share_link

checkpoint = args.checkpoint
test_file = args.input_file
print(test_file)

# Change this line if you have POS tags in your data, eg.- {0: 'text', 1:'pos', 2:'ner'}
columns = {0: 'text', 1:'ner'}

data_folder = input_folder

tag_type = 'ner'

corpus: Corpus = ColumnCorpus(data_folder, columns, train_file=args.dev_file, dev_file=args.dev_file, test_file=test_file, column_delimiter="\t")

tag_dictionary = corpus.make_label_dictionary(label_type=tag_type)

tagger: SequenceTagger = SequenceTagger.load(checkpoint)

trainer: ModelTrainer = ModelTrainer(tagger, corpus)

trainer.final_test(output_folder,eval_mini_batch_size=256,main_evaluation_metric=('micro avg', 'f1-score'))
