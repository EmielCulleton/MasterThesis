import config
import torch
import math
from transformers import BertTokenizer, BertForTokenClassification

class BERTModel:
    def __init__(self, config):
        self.config = config

        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-cased", do_lower_case=False
        )

        self.model = BertForTokenClassification.from_pretrained(
            "bert-base-cased",
            num_labels=self.config.num_classes,
            output_attentions=False,
            output_hidden_states=False,
        )