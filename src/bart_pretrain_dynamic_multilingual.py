import numpy as np
import shutil
import nltk
import copy
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, PreTrainedTokenizerBase
from datasets import load_metric, load_dataset
import transformers
transformers.logging.set_verbosity_info()
from typing import Any,Optional,Union
from enum import Enum
from dataclasses import dataclass
from utils import mask_entities, mask_words, get_random_gauss_value
import random
import torch


import argparse
parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument('--mask_entities', default='False', help='Should we mask the entities following the gaussian distribution')
parser.add_argument('-a', '--mask_attn', default='all', help='Which attn mode to select (all/gauss/none)')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size while training')
parser.add_argument('--mode','-m', default='attn', help='masking mode (both/either/attn)')
parser.add_argument('--directory','-dir', default='attn', help='data directory where train and dev files are located')
parser.add_argument('--train_file','-tf', default='attn', help='train file name')
parser.add_argument('--dev_file','-df', default='attn', help='dev file name')
parser.add_argument('--file_name','-f', type=str, default='', help='file name for output')
parser.add_argument('--lang','-la', type=str, default='', help='language you are working on')
parser.add_argument('--seed', '-s', type=int, default='-1', help='random seed')

args = parser.parse_args()

if not args.seed==-1:
    transformers.set_seed(args.seed)
    torch.backends.cudnn.deterministic = True

args.mask_entities = False if args.mask_entities=='False' else True

print(args)

# load the preprocessed dataset with the four kinds of sketches
data_files = {"train": args.train_file+'.csv', "validation":args.dev_file+'.csv'}
tokenized_dataset = load_dataset(args.directory, data_files=data_files)
print(tokenized_dataset)

# define the inputs and labels for sketch-based reconstruction pre-training
max_input_length = 75
max_target_length = 75

# pretrained checkpoint:
model_checkpoint = "facebook/mbart-large-50"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint,src_lang=args.lang, tgt_lang=args.lang)

# new tokens
new_tokens = ['<b-corp>', '<b-cw>', '<b-grp>', '<b-loc>', '<b-per>', '<b-prod>', '<i-corp>', '<i-cw>', '<i-grp>', '<i-loc>', '<i-per>', '<i-prod>']

# check if the tokens are already in the vocabulary
new_tokens = set(new_tokens) - set(tokenizer.vocab.keys())

# add the tokens to the tokenizer vocabulary
tokenizer.add_tokens(list(new_tokens), special_tokens=True)

class ExplicitEnum(str, Enum):
    """
    Enum with more explicit error message for missing values.
    """

    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            f"{value} is not a valid {cls.__name__}, please select one of {list(cls._value2member_map_.keys())}"
        )


class PaddingStrategy(ExplicitEnum):
    """
    Possible values for the `padding` argument in [`PreTrainedTokenizerBase.__call__`]. Useful for tab-completion in an
    IDE.
    """

    LONGEST = "longest"
    MAX_LENGTH = "max_length"
    DO_NOT_PAD = "do_not_pad"


@dataclass
class DataCollatorForSeq2Seq:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.
    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        model ([`PreTrainedModel`]):
            The model that is being trained. If set and has the *prepare_decoder_input_ids_from_labels*, use it to
            prepare the *decoder_input_ids*
            This is useful when using *label_smoothing* to avoid calculating loss twice.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single sequence
              is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
              lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        # print(features)
        # features= features[:2]
        text = [i['text'] for i in features]
        bert_attn = [i['bert_att'] for i in features]
        labels = [i['label'] for i in features]

        # c = list(zip(text, bert_attn, labels))
        # random.shuffle(c)
        # text, bert_attn, labels = zip(*c)
        # text, bert_attn, labels = list(text), list(bert_attn), list(labels)

        sketch = []
        n_text = []

        for i in range(len(text)): # for ever datapoint in a batch
            # print('Text: ', text[i])
            new_text, new_bert_attn, new_label = text[i].split(), bert_attn[i].split(), labels[i].split()
            assert len(new_text) == len(new_bert_attn) == len(new_label)
            copy_text = copy.deepcopy(new_text)

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

            copy_text = ' '.join(mask_entities(copy_text, new_label, False))
            n_text.append(copy_text)
            sketch.append(new_sketch)
            # print('Text: ', copy_text)
            # print('Sketch: ', new_sketch, '\n\n')

        # print(sketch,'\n',n_text)
        # fdafddfa

        model_inputs = tokenizer(sketch, max_length=max_input_length, truncation=True)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(n_text, max_length=max_target_length, truncation=True)
        model_inputs['labels'] = labels['input_ids']

        features = []
        for i in range(len(model_inputs['labels'])):
            features.append({'input_ids': model_inputs['input_ids'][i],
                            'attention_mask': model_inputs['attention_mask'][i],
                            'labels': model_inputs['labels'][i] })

        del model_inputs, labels, sketch, text, bert_attn

        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        # prepare decoder_input_ids
        if (
            labels is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids

        return features


def compute_metrics(eval_pred):
    return {}





##################################################################
#                     training
##################################################################

batch_size = args.batch_size
num_train_epochs = args.epochs
model_name = model_checkpoint.split("/")[-1]

# load the pretrained weights
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
# load only the model, without weights
# config = AutoConfig.from_pretrained(model_checkpoint)
# model =  AutoModel.from_config(config)

# add new, random embeddings for the new tokens
model.resize_token_embeddings(len(tokenizer))

with torch.no_grad():
    # new_tokens = ['<b-corp>', '<b-cw>', '<b-grp>', '<b-loc>', '<b-per>', '<b-prod>', '<i-corp>', '<i-cw>', '<i-grp>', '<i-loc>', '<i-per>', '<i-prod>']
    model.model.encoder.embed_tokens.weight[-1, :] += model.model.encoder.embed_tokens.weight[57877, :]
    model.model.encoder.embed_tokens.weight[-2, :] += model.model.encoder.embed_tokens.weight[30816, :]
    model.model.encoder.embed_tokens.weight[-3, :] += model.model.encoder.embed_tokens.weight[87632, :]
    model.model.encoder.embed_tokens.weight[-4, :] += model.model.encoder.embed_tokens.weight[51588, :]
    model.model.encoder.embed_tokens.weight[-5, :] += model.model.encoder.embed_tokens.weight[85403, :]
    model.model.encoder.embed_tokens.weight[-6, :] += model.model.encoder.embed_tokens.weight[216487, :]
    model.model.encoder.embed_tokens.weight[-7, :] += model.model.encoder.embed_tokens.weight[57877, :]
    model.model.encoder.embed_tokens.weight[-8, :] += model.model.encoder.embed_tokens.weight[30816, :]
    model.model.encoder.embed_tokens.weight[-9, :] += model.model.encoder.embed_tokens.weight[87632, :]
    model.model.encoder.embed_tokens.weight[-10, :] += model.model.encoder.embed_tokens.weight[51588, :]
    model.model.encoder.embed_tokens.weight[-11, :] += model.model.encoder.embed_tokens.weight[85403, :]
    model.model.encoder.embed_tokens.weight[-12, :] += model.model.encoder.embed_tokens.weight[216487, :]
    model.model.decoder.embed_tokens.weight[-1, :] += model.model.decoder.embed_tokens.weight[57877, :]
    model.model.decoder.embed_tokens.weight[-2, :] += model.model.decoder.embed_tokens.weight[30816, :]
    model.model.decoder.embed_tokens.weight[-3, :] += model.model.decoder.embed_tokens.weight[87632, :]
    model.model.decoder.embed_tokens.weight[-4, :] += model.model.decoder.embed_tokens.weight[51588, :]
    model.model.decoder.embed_tokens.weight[-5, :] += model.model.decoder.embed_tokens.weight[85403, :]
    model.model.decoder.embed_tokens.weight[-6, :] += model.model.decoder.embed_tokens.weight[216487, :]
    model.model.decoder.embed_tokens.weight[-7, :] += model.model.decoder.embed_tokens.weight[57877, :]
    model.model.decoder.embed_tokens.weight[-8, :] += model.model.decoder.embed_tokens.weight[30816, :]
    model.model.decoder.embed_tokens.weight[-9, :] += model.model.decoder.embed_tokens.weight[87632, :]
    model.model.decoder.embed_tokens.weight[-10, :] += model.model.decoder.embed_tokens.weight[51588, :]
    model.model.decoder.embed_tokens.weight[-11, :] += model.model.decoder.embed_tokens.weight[85403, :]
    model.model.decoder.embed_tokens.weight[-12, :] += model.model.decoder.embed_tokens.weight[216487, :]


# logging_steps = len(tokenized_dataset['train']) // batch_size
if args.directory[-1]!='/':
    args.directory += '/'

output_dir = f"{args.directory}{args.train_file}-{args.file_name}"

training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="epoch",
    save_strategy = 'epoch',
    save_total_limit = 1,
    load_best_model_at_end = True,
    metric_for_best_model = "eval_loss",
    fp16 = True,
    learning_rate=5.6e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    num_train_epochs=num_train_epochs,
    predict_with_generate=True,
    logging_steps=60,
    remove_unused_columns=False
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)


trainer = Seq2SeqTrainer(
    model,
    training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)


trainer.train()
save_path = output_dir+"-final"
trainer.save_model(save_path)

shutil.rmtree(output_dir)