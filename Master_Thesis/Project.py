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
