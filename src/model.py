import torch
import torch.nn as nn
from transformers import *

class WrimeBert(nn.Module):
    def __init__(self, model_type, tokenizer, n_label):
        super(WrimeBert, self).__init__()

        bert_conf = BertConfig(model_type)
        bert_conf.vocab_size = tokenizer.vocab_size

        self.bert = AutoModel.from_pretrained(model_type, config=bert_conf)
        self.fc = nn.Linear(bert_conf.hidden_size, n_label)


    def forward(self, ids, mask, token_type_ids):
        _, h = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        h = self.fc(h)
        return h
