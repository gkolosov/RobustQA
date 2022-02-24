import os
from transformers import DistilBertForQuestionAnswering
from transformers import AdamW

from abstract_trainer import AbstractTrainer, main


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
    main(BaselineTrainer)
