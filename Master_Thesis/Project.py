#importing necessary packages
import pandas as pd                                                     #pandas are dope

import numpy as np
from numpy.random import seed as np_seed; np_seed(36)

import sklearn as sk
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

#import matplotlib.pyplot as plt
import json
import tarfile                                                          #read and write tar archive files. Enables the use of gzip, bz2 and lzma compression. Use also yet to be determined
import os; os.environ["PYTHONHASHSEED"] = str(36)                       #misc. operating system interfaces, use yet to be determined
import bs4                                                              #loading xml documents
import lxml                                                             #loading xml documents
import time                                                             #check running time CPU/GPU during process -> use for time.process_time()
import requests                                                         #needed for testing import_xml
import copy
import sys

import torch                                                            #ML framework. FUN
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
from torch.utils.data import SequentialSampler
torch.manual_seed(36)

import io                                                               #allows us to manage the file-related input and output operations
import random
from random import seed; seed(36)

from tqdm import tqdm, trange                                           #loads in progress bar

import transformers
from transformers import BertTokenizer                                  # run end-to-end tokenization: punctuation splitting + word piece
from transformers import BertConfig                                     # configuration class
from transformers import AdamW                                          # implements Adam learning rate optimization algorithm
from transformers import BertForTokenClassification
from transformers import BertForSequenceClassification                  # BERT Model transformer with a sequence classification/regression head on top
from transformers import get_linear_schedule_with_warmup                # creates a schedule with a learning rate that decreases linearly after linearly increasing during a warm-up period
from transformers import pipeline

from bs4 import BeautifulSoup as bs 

#---
# Define a few standard items:

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")




#---

def print_soup(soup_content):
    file = open(soup_content)
    contents = file.read()
    soup = bs(contents, features="xml")
    print(soup.head)




