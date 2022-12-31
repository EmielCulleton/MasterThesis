import transformers
import pandas as pd
import csv
import json
from transformers import pipeline
from transformers import BertTokenizer, BertModel, BertForMaskedLM
from transformers import AutoTokenizer, AutoModelForTokenClassification
import dataset
from datasets import Dataset
from datasets import load_dataset


MAX_LEN = 750
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 8
EPOCHS = 10
BASE_MODEL_PATH = " "
MODEL_PATH = "model.bin"

with open("/home/emiel/data/textwash_data.json", "r") as json_file_open:
    TRAINING_FILE = json.load(json_file_open)


# TOKENIZER = AutoTokenizer.from_pretrained("bert-base-uncased") #xlm-roberta-large
# MODEL = AutoTokenizer.from_pretrained("bert-base-uncased")
# # CLASSIFIER = pipeline('ner', model=MODEL, tokenizer=TOKENIZER)