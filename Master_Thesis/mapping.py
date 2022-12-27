from transformers import AutoTokenizer
import json
import pandas as pd
from datasets import Dataset
from collections import defaultdict
import requests
import datetime
from tqdm import tqdm

#extracts the information from the trainining file in order to feed it to the model for fine-tuning


tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

with open("/home/emiel/data/textwash_data.json", "r") as json_file_open:
    json_file = json.load(json_file_open)

# print(json_file[0])


def convert_ents_to_bio(tokens, spans):
    tags = ["O"] * len(tokens)
    for span in tqdm(spans):
        for start, end, label in span["spans"]:
            start, end, label = span["token_start"], span["token_end"], span["label"]
            if start == end:
                tags[start] = 'S-'+ label
            else:
                tags[start] = 'B-' + label
                tags[start+1: end + 1] = ['I-'+label]*(end - start)
    return tags


#extracting individual tokens from the imput file
def extract_words(input):
    words = []
    for entry in tqdm(input): 
        for dict in entry["tokens"]:
            words.append(list(dict.values())[0])

    return words



ts = convert_ents_to_bio(json_file[0]["tokens"], json_file[0]["spans"])
words = extract_words(json_file)
print(len(ts), len(words))


