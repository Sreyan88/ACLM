import flair
import torch
import argparse
import os

parser = argparse.ArgumentParser(description='Train flair model')
parser.add_argument('--input_folder', '-i', help='Name of the input folder containing train, dev and test files')
parser.add_argument('--output_folder', '-o', help='Name of the output folder')
parser.add_argument('--gpu', '-g', help='Use gpu/cpu, put "cuda" if gpu and "cpu" if cpu')
parser.add_argument('--train_file', '-tf', help='train file name')
parser.add_argument('--batch_size', '-bs',type=int, help='batch-size')
parser.add_argument('--lr', '-l',type=float, help='learning rate')
parser.add_argument('--epochs', '-ep',type=int, help='epochs')
parser.add_argument('--language', '-lang', help='language short code (file prefix)')
parser.add_argument('--seed', '-s', type=int, help='random seed')
args = parser.parse_args()

print(args)

flair.set_seed(args.seed)
torch.backends.cudnn.deterministic = True

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
from flair.embeddings import TokenEmbeddings, StackedEmbeddings
from flair.embeddings import TransformerWordEmbeddings

# Change this line if you have POS tags in your data, eg.- {0: 'text', 1:'pos', 2:'ner'}
columns = {0: 'text', 1:'ner'}

data_folder = input_folder

tag_type = 'ner'

corpus: Corpus = ColumnCorpus(data_folder, columns, train_file=args.train_file,
                              dev_file=f'{args.language}_dev.conll',test_file=f'{args.language}_test.conll',column_delimiter="\t", comment_symbol="# id")

tag_dictionary = corpus.make_label_dictionary(label_type=tag_type)

embedding_types: List[TokenEmbeddings] = [
    TransformerWordEmbeddings('xlm-roberta-large',fine_tune = True,model_max_length=256),
 ]

embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)
tagger: SequenceTagger = SequenceTagger(use_rnn = False,
                                        embeddings=embeddings,
                                        tag_dictionary=tag_dictionary,
                                        tag_type=tag_type,
                                        use_crf=False)

trainer: ModelTrainer = ModelTrainer(tagger, corpus)

trainer.train(output_folder, learning_rate=args.lr,save_final_model=False,
             mini_batch_size=args.batch_size,
             max_epochs=args.epochs,embeddings_storage_mode='gpu',main_evaluation_metric=('micro avg', 'f1-score'), shuffle=True)