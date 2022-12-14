#importing necessary packages

import pandas as pd                                                     #pandas are dope
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
import tarfile                                                          #read and write tar archive files. Enables the use of gzip, bz2 and lzma compression. Use also yet to be determined
import os; os.environ["PYTHONHASHSEED"] = str(36)                       #misc. operating system interfaces, use yet to be determined
import bs4                                                              #loading xml documents
import lxml                                                             #loading xml documents
import time                                                             #check running time CPU/GPU during process -> use for time.process_time()
import requests                                                         #needed for testing import_xml
import torch                                                            #ML framework. FUN
import torch.nn.functional as F
import io                                                               #allows us to manage the file-related input and output operations
import random
#matplotlib inline

from tqdm import tqdm, trange                                           #loads in progress bar
from random import seed; seed(36)
from numpy.random import seed as np_seed; np_seed(36)
from transformers import BertTokenizer, BertConfig,AdamW, BertForSequenceClassification, BertForTokenClassification, get_linear_schedule_with_warmup, pipeline
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from bs4 import BeautifulSoup as bs

#---
# Define a few standard items:

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

#---

content = []
# Read the XML file
with open("sample.xml", "r") as file:
    # Read each line in the file, readlines() returns a list of lines
    content = file.readlines()
    # Combine the lines in the list into a string
    content = "".join(content)
    bs_content = bs(content, "lxml")


#print(content)

url = 'https://rss.nytimes.com/services/xml/rss/nyt/US.xml'
xml_data = requests.get(url).content 

def parse_xml(xml_data):
  # Initializing soup variable
    soup = bs(xml_data, 'xml')

  # Creating column for table
    df = pd.DataFrame(columns=['guid', 'title', 'pubDate', 'description'])

  # Iterating through item tag and extracting elements
    all_items = soup.find_all('item')
    items_length = len(all_items)
    
    for index, item in enumerate(all_items):
        guid = item.find('guid').text
        title = item.find('title').text
        pub_date = item.find('pubDate').text
        description = item.find('description').text

       # Adding extracted elements to rows in table
        row = {
            'guid': guid,
            'title': title,
            'pubDate': pub_date,
            'description': description
        }

        df = df.append(row, ignore_index=True)
        print(f'Appending row %s of %s' % (index+1, items_length))

    return df

df = parse_xml(xml_data)
#print(df.head(5))

with open("cow_sents.txt", "r") as cow:
    content = cow.readlines()
    # Combine the lines in the list into a string
    content = "".join(content)
    bs_content = bs(content, "lxml")