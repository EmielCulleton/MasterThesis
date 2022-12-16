import transformers
import pandas as pd
import csv
import json
from transformers import pipeline
from transformers import BertTokenizer, BertModel, BertForMaskedLM

MAX_LEN = 128
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 8
EPOCHS = 10
BASE_MODEL_PATH = " "
MODEL_PATH = "model.bin"
file = open("/home/emiel/data/textwash_data.json")
TRAINING_FILE = json.load(file)
TOKENIZER = BertTokenizer.from_pretrained("bert-base-uncased",
    do_lower_case=True
    )
model = BertForMaskedLM.from_pretrained("bert-base-uncased")
unmasker = pipeline('fill-mask', model = 'bert-base-uncased')