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


MAX_LEN = 128
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 8
EPOCHS = 10
BASE_MODEL_PATH = " "
MODEL_PATH = "model.bin"



#directly importing it as json didn't work might improve this later
file = pd.read_json("/home/emiel/data/textwash_data.json")
training_file_pandas = pd.DataFrame(file)
TRAINING_FILE = Dataset.from_pandas(training_file_pandas)
print(TRAINING_FILE)

train_test_split = TRAINING_FILE.train_test_split(test_size=0.2, shuffle=True)
#print(train_test_split)

#data = load_dataset("json", data_files="/home/emiel/data/textwash_data.json", field="text")


TOKENIZER = AutoTokenizer.from_pretrained("xlm-roberta-large")
MODEL = AutoModelForTokenClassification.from_pretrained("xlm-roberta-large")
CLASSIFIER = pipeline('ner', model=MODEL, tokenizer=TOKENIZER)

CLASSIFIER("Richard saw his friend through the glass window in London valley")

