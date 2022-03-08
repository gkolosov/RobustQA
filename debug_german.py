import os
from collections import OrderedDict
import torch
import util

#from tensorboardX import SummaryWriter

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


    #train_loader = DataLoader(train_dataset,
    #                          batch_size=args.batch_size,
    #                          sampler=RandomSampler(train_dataset))
    #val_loader = DataLoader(val_dataset,
    #                        batch_size=args.batch_size,
    #                        sampler=SequentialSampler(val_dataset))
    #best_scores = trainer.train(model, train_loader, val_loader, val_dict)

if __name__ == '__main__':
    main()