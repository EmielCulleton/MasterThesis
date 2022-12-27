import transformers
import pandas as pd
import csv
import json
from transformers import pipeline
from transformers import BertTokenizer, BertModel, BertForMaskedLM
from transformers import AutoTokenizer, AutoModelForTokenClassification

MAX_LEN = 128
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 8
EPOCHS = 10
BASE_MODEL_PATH = " "
MODEL_PATH = "model.bin"
file = pd.read_json("/home/emiel/data/textwash_data.json")
TRAINING_FILE = pd.DataFrame(file)
TOKENIZER = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
unmasker = pipeline('fill-mask', model = 'bert-base-uncased')
nlp = pipeline("ner", model=model, tokenizer=TOKENIZER)