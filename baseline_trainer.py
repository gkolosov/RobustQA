import json
import os
import torch
import csv
import util
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForQuestionAnswering
from transformers import AdamW


from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from args import get_train_test_args

from abstract_trainer import AbstractTrainer, get_dataset


class BaselineTrainer(AbstractTrainer):
    def __init__(self, args, log):
        super(BaselineTrainer, self).__init__(args, log)

    def setup_model(self, args, do_train=False, do_eval=False):
        model = None
        if do_train:
            model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")
        if do_eval:
            checkpoint_path = os.path.join(args.save_dir, 'checkpoint')
            model = DistilBertForQuestionAnswering.from_pretrained(checkpoint_path)
            model.to(args.device)

        return model

    def setup_model_optim(self, model):
        device = self.device
        model.to(device)
        optim = AdamW(model.parameters(), lr=self.lr)
        return device, optim

    def step(self, batch, device, model, optim):
        optim.zero_grad()
        model.train()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        start_positions = batch['start_positions'].to(device)
        end_positions = batch['end_positions'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask,
                        start_positions=start_positions,
                        end_positions=end_positions)
        loss = outputs[0]
        loss.backward()
        optim.step()
        return input_ids, loss

    def save(self, model):
        model.save_pretrained(self.path)


if __name__ == "__main__":
    # define parser and arguments
    args = get_train_test_args()

    util.set_seed(args.seed)
    model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    if args.do_train:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        args.save_dir = util.get_save_dir(args.save_dir, args.run_name)
        log = util.get_logger(args.save_dir, 'log_train')
        log.info(f'Args: {json.dumps(vars(args), indent=4, sort_keys=True)}')
        log.info("Preparing Training Data...")
        args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        trainer = BaselineTrainer(args, log)
        model = trainer.setup_model(args, do_train=True)
        train_dataset, _ = get_dataset(args, args.train_datasets, args.train_dir, tokenizer, 'train')
        log.info("Preparing Validation Data...")
        val_dataset, val_dict = get_dataset(args, args.train_datasets, args.val_dir, tokenizer, 'val')
        train_loader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  sampler=RandomSampler(train_dataset))
        val_loader = DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                sampler=SequentialSampler(val_dataset))
        best_scores = trainer.train(model, train_loader, val_loader, val_dict)
    if args.do_eval:
        args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        split_name = 'test' if 'test' in args.eval_dir else 'validation'
        log = util.get_logger(args.save_dir, f'log_{split_name}')
        trainer = BaselineTrainer(args, log)
        model = trainer.setup_model(args, do_eval=True)
        eval_dataset, eval_dict = get_dataset(args, args.eval_datasets, args.eval_dir, tokenizer, split_name)
        eval_loader = DataLoader(eval_dataset,
                                 batch_size=args.batch_size,
                                 sampler=SequentialSampler(eval_dataset))
        eval_preds, eval_scores = trainer.evaluate(model, eval_loader,
                                                   eval_dict, return_preds=True,
                                                   split=split_name)
        results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in eval_scores.items())
        log.info(f'Eval {results_str}')
        # Write submission file
        sub_path = os.path.join(args.save_dir, split_name + '_' + args.sub_file)
        log.info(f'Writing submission file to {sub_path}...')
        with open(sub_path, 'w', newline='', encoding='utf-8') as csv_fh:
            csv_writer = csv.writer(csv_fh, delimiter=',')
            csv_writer.writerow(['Id', 'Predicted'])
            for uuid in sorted(eval_preds):
                csv_writer.writerow([uuid, eval_preds[uuid]])