import os
import torch
from transformers import AdamW

from model_advers import DomainQA

from abstract_trainer import AbstractTrainer, main


class AdversarialTrainer(AbstractTrainer):
    def __init__(self, args, log):
        super(AdversarialTrainer, self).__init__(args, log)

    def setup_model(self, args, do_train=False, do_eval=False):
        model = DomainQA(num_classes=6, hidden_size=768,
                         num_layers=3, dropout=0.1, dis_lambda=0.5, concat=False, anneal=False)
        if do_eval:
            checkpoint_path = os.path.join(args.save_dir, 'checkpoint')
            model.load_state_dict(torch.load(checkpoint_path))

        if args.freeze_bert:
            for param in model.bert.parameters():
                param.requires_grad = False

        return model

    def setup_model_optim(self, model: DomainQA):
        device = self.device
        model.to(device)
        optim = dict()
        qa_params = list(model.qa_outputs.parameters()) + list(model.bert.parameters())
        optim['qa'] = AdamW(qa_params, lr=self.lr)
        dis_params = model.discriminator.parameters()
        optim['dis'] = AdamW(dis_params, lr=self.lr)

        return device, optim

    def step(self, batch, device, model: DomainQA, optim):
        """
        Step adversarial model
        """

        model.train()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        start_positions = batch['start_positions'].to(device)
        end_positions = batch['end_positions'].to(device)
        # TODO: token_type_ids is note necessary. Original code used it for pytorch_pretrained_bert import BertModel.
        token_type_ids = batch['sequence_ids'].to(device)
        labels = batch['labels'].to(device)

        optim['qa'].zero_grad()
        qa_loss = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                        start_positions=start_positions,
                        end_positions=end_positions, dtype='qa', labels=labels)

        qa_loss.backward()
        optim['qa'].step()

        optim['dis'].zero_grad()
        dis_loss = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                         start_positions=start_positions,
                         end_positions=end_positions, dtype='dis', labels=labels)
        dis_loss.backward()
        optim['dis'].step()

        return input_ids, qa_loss

    def save(self, model):
        torch.save(model.qa_outputs.state_dict(), self.path)


if __name__ == "__main__":
    main(AdversarialTrainer)
