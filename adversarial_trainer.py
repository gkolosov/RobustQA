import os
import torch
from transformers import AdamW

from model_advers import DomainQA

from abstract_trainer import AbstractTrainer, main


class AdversarialTrainer(AbstractTrainer):
    def __init__(self, args, log):
        super(AdversarialTrainer, self).__init__(args, log)

    def setup_model(self, args, do_train=False, do_eval=False, num_classes=3):
        model = DomainQA(num_classes=num_classes, hidden_size=768,
                         num_layers=3, dropout=0.1, dis_lambda=args.dis_lambda, concat=False, anneal=False)

        if do_eval or args.load_dir != '':
            if args.load_dir != '':
                checkpoint_path = os.path.join(args.load_dir, 'checkpoint')
            else:
                checkpoint_path = os.path.join(args.save_dir, 'checkpoint')
            print('Loading parameters from: %s' % checkpoint_path)
            state_dict_path = os.path.join(checkpoint_path, 'model.pt')
            model_state_dict = torch.load(state_dict_path, map_location=self.device)
            model.qa_outputs.load_state_dict(model_state_dict['qa_outputs'])
            model.discriminator.load_state_dict(model_state_dict['discriminator'])
            model.bert.load_state_dict(model_state_dict['bert'])

        if args.freeze_bert:
            for param in model.bert.parameters():
                param.requires_grad = False

        if args.freeze_dis:
            for param in model.discriminator.parameters():
                param.requires_grad = False

        device = self.device
        model.to(device)

        return model

    def setup_model_optim(self, model: DomainQA):
        optim = dict()
        qa_params = list(model.qa_outputs.parameters()) + list(model.bert.parameters())
        optim['qa'] = AdamW(qa_params, lr=self.lr)
        dis_params = model.discriminator.parameters()
        optim['dis'] = AdamW(dis_params, lr=self.lr)

        return optim

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
        qa_loss = model(input_ids=input_ids, attention_mask=attention_mask,
                        start_positions=start_positions,
                        end_positions=end_positions, dtype='qa', labels=labels)

        qa_loss.backward()
        optim['qa'].step()

        optim['dis'].zero_grad()
        dis_loss = model(input_ids=input_ids, attention_mask=attention_mask,
                         start_positions=start_positions,
                         end_positions=end_positions, dtype='dis', labels=labels)
        dis_loss.backward()
        optim['dis'].step()

        return input_ids, (qa_loss, dis_loss)

    def save(self, model: DomainQA):
        state_dict_path = os.path.join(self.path, 'model.pt')
        torch.save({'qa_outputs': model.qa_outputs.state_dict(),
                    'discriminator': model.discriminator.state_dict(),
                    'bert': model.bert.state_dict()},
                   state_dict_path)


if __name__ == "__main__":
    main(AdversarialTrainer)
