import os
from collections import OrderedDict
import torch
import util
import pandas as pd
#from tensorboardX import SummaryWriter
from data_augmentation import synonym_replacement
from tqdm import tqdm

import json
import csv
from transformers import DistilBertTokenizerFast


from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from args import get_train_test_args

from abstract_trainer import get_dataset


def main():
    args = get_train_test_args()

    util.set_seed(args.seed)
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    args.save_dir = util.get_save_dir(args.save_dir, args.run_name)
    log = util.get_logger(args.save_dir, 'log_train')
    log.info(f'Args: {json.dumps(vars(args), indent=4, sort_keys=True)}')
    log.info("Preparing Training Data...")
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    #trainer = trainer_cls(args, log)
    #train_dataset, _, _ = get_dataset(args, args.train_datasets, args.train_dir, tokenizer, 'train', debug=args.debug)
    #model = trainer.setup_model(args, do_train=True)
    log.info("Preparing Validation Data...")
    val_dataset, val_dict, _ = get_dataset(args, args.train_datasets, args.val_dir, tokenizer, 'val', debug=args.debug)




def sr():
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    datasets = 'duorc,race'
    data_dir = 'datasets/oodomain_train'

    datasets = datasets.split(',')
    dataset_dict = None
    dataset_name = ''
    label = 0
    for dataset in datasets:
        dataset_name += f'_{dataset}'
        dataset_dict_curr = util.read_squad(f'{data_dir}/{dataset}')
        dataset_dict_curr['label'] = label
        dataset_dict = util.merge(dataset_dict, dataset_dict_curr)
        label += 1
    # data_encodings = read_and_process(args, tokenizer, dataset_dict, data_dir, dataset_name, split_name)

    df = pd.DataFrame({x: dataset_dict[x] for x in dataset_dict if x not in ['label']})
    df['start_char'] = df.answer.apply(lambda x: x['answer_start'][0])
    df['end_char'] = df['start_char'] + df.answer.apply(lambda x: len(x['text'][0]))
    df['final_answer'] = [A[B:C] for A, B, C in zip(df.context, df['start_char'], df['end_char'])]

    line = df.loc[20, "context"].split('\n')[0]
    #clean_line = get_only_chars(line).split(" ")
    clean_line = line.split(" ")
    modif_line = synonym_replacement(clean_line, 20)
    print(line)
    print(modif_line)

if __name__ == '__main__':
    sr()