import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# from pytorch_pretrained_bert import BertModel, BertConfig
from transformers import DistilBertModel

# Adapted from https://github.com/seanie12/mrqa


def kl_coef(i):
    # coef for KL annealing
    # reaches 1 at i = 22000
    # https://github.com/kefirski/pytorch_RVAE/blob/master/utils/functional.py
    return (math.tanh((i - 3500) / 1000) + 1) / 2


class DomainDiscriminator(nn.Module):
    def __init__(self, num_classes=3, input_size=768 * 2,
                 hidden_size=768, num_layers=3, dropout=0.1):
        super(DomainDiscriminator, self).__init__()
        self.num_layers = num_layers
        hidden_layers = []
        for i in range(num_layers):
            if i == 0:
                input_dim = input_size
            else:
                input_dim = hidden_size
            hidden_layers.append(nn.Sequential(
                nn.Linear(input_dim, hidden_size),
                nn.ReLU(), nn.Dropout(dropout)
            ))
        hidden_layers.append(nn.Linear(hidden_size, num_classes))
        self.hidden_layers = nn.ModuleList(hidden_layers)

    def forward(self, x):
        # forward pass
        for i in range(self.num_layers - 1):
            x = self.hidden_layers[i](x)
        logits = self.hidden_layers[-1](x)
        log_prob = F.log_softmax(logits, dim=1)
        return log_prob


class DomainQA(nn.Module):
    def __init__(self, checkpoint_path=None, num_classes=3, hidden_size=768,
                 num_layers=3, dropout=0.1, dis_lambda=0.5, concat=False, anneal=False):
        super(DomainQA, self).__init__()

        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")

        self.config = self.bert.config

        self.qa_outputs = nn.Linear(hidden_size, 2)
        # init weight
        self.qa_outputs.weight.data.normal_(mean=0.0, std=0.02)
        self.qa_outputs.bias.data.zero_()
        if concat:
            input_size = 2 * hidden_size
        else:
            input_size = hidden_size
        self.discriminator = DomainDiscriminator(num_classes, input_size, hidden_size, num_layers, dropout)

        self.num_classes = num_classes
        self.dis_lambda = dis_lambda
        self.anneal = anneal
        self.concat = concat
        self.sep_id = 102

    def forward(self, input_ids, attention_mask,
                start_positions=None, end_positions=None, labels=None,
                dtype=None, global_step=22000, return_dict=None):
        if dtype == "qa":
            qa_loss = self.forward_qa(input_ids, attention_mask,
                                      start_positions, end_positions, global_step)
            return qa_loss
        elif dtype == "dis":
            assert labels is not None
            dis_loss = self.forward_discriminator(input_ids, attention_mask, labels)
            return dis_loss
        else:
            last_hidden_state, = self.bert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=False, return_dict=False)
            # TODO: Is this correct
            logits = self.qa_outputs(last_hidden_state)
            end_logits, start_logits = self.compute_segment_logits(logits)

            return start_logits, end_logits

    def compute_segment_logits(self, logits):
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        return end_logits, start_logits

    def forward_qa(self, input_ids, attention_mask, start_positions, end_positions, global_step):
        last_hidden_state, = self.bert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=False, return_dict=False)
        cls_embedding = last_hidden_state[:, 0]  # [b, d] : [CLS] representation
        if self.concat:
            sep_embedding = self.get_sep_embedding(input_ids, last_hidden_state)
            hidden = torch.cat([cls_embedding, sep_embedding], dim=1)
        else:
            hidden = cls_embedding
        log_prob = self.discriminator(hidden)
        targets = torch.ones_like(log_prob) * (1 / self.num_classes)
        # As with NLLLoss, the input given is expected to contain log-probabilities
        # and is not restricted to a 2D Tensor. The targets are given as probabilities
        kl_criterion = nn.KLDivLoss(reduction="batchmean")
        if self.anneal:
            self.dis_lambda = self.dis_lambda * kl_coef(global_step)
        # TODO: check inputs
        kld = self.dis_lambda * kl_criterion(log_prob, targets)

        logits = self.qa_outputs(last_hidden_state)
        end_logits, start_logits = self.compute_segment_logits(logits)

        # If we are on multi-GPU, split add a dimension
        if len(start_positions.size()) > 1:
            start_positions = start_positions.squeeze(-1)
        if len(end_positions.size()) > 1:
            end_positions = end_positions.squeeze(-1)
        # sometimes the start/end positions are outside our model inputs, we ignore these terms
        ignored_index = start_logits.size(1)
        start_positions.clamp_(0, ignored_index)
        end_positions.clamp_(0, ignored_index)

        loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
        start_loss = loss_fct(start_logits, start_positions)
        end_loss = loss_fct(end_logits, end_positions)
        qa_loss = (start_loss + end_loss) / 2
        total_loss = qa_loss + kld
        return total_loss

    def forward_discriminator(self, input_ids, attention_mask, labels):
        with torch.no_grad():
            last_hidden_state, = self.bert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=False, return_dict=False)
            cls_embedding = last_hidden_state[:, 0]  # [b, d] : [CLS] representation
            if self.concat:
                sep_embedding = self.get_sep_embedding(input_ids, last_hidden_state)
                hidden = torch.cat([cls_embedding, sep_embedding], dim=-1)  # [b, 2*d]
            else:
                hidden = cls_embedding
        log_prob = self.discriminator(hidden.detach())
        criterion = nn.NLLLoss()
        loss = criterion(log_prob, labels)

        return loss

    def get_sep_embedding(self, input_ids, sequence_output):
        batch_size = input_ids.size(0)
        sep_idx = (input_ids == self.sep_id).sum(1)
        sep_embedding = sequence_output[torch.arange(batch_size), sep_idx]
        return sep_embedding
