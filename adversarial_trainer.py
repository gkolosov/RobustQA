import os
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForQuestionAnswering
from transformers import AdamW

from abstract_trainer import AbstractTrainer, get_dataset


class AdversarialTrainer(AbstractTrainer):
    def __init__(self, args, log):
        super(AdversarialTrainer, self).__init__(args, log)

    def setup_model(self, args, do_train=False, do_eval=False):
        model = None
        if do_train:
            # TODO: create model for training
            raise NotImplementedError
        if do_eval:
            # TODO: load model for evaluation
            raise NotImplementedError

        return model

    def setup_model_optim(self, model):
        """
        Setup adversarial model
        """
        raise NotImplementedError

    def step(self, batch, device, model, optim):
        """
        Step adversarial model
        """
        raise NotImplementedError


if __name__ == "__main__":
    pass
