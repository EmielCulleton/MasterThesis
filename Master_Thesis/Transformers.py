#importing necessary packages

import pandas as pd                                                     #pandas are dope
import numpy as np
import sklearn as sk
import tarfile                                                          #read and write tar archive files. Enables the use of gzip, bz2 and lzma compression. Use also yet to be determined
import os                                                               #misc. operating system interfaces, use yet to be determined
import bs4                                                              #loading xml documents
import lxml                                                             #loading xml documents
import time                                                             #check running time CPU/GPU during process -> use for time.process_time()

from tqdm import tqdm                                                   #loads in progress bar
from transformers import BertTokenizer
from transformers import pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from bs4 import BeautifulSoup as bs

#---

content = []
# Read the XML file
with open("sample.xml", "r") as file:
    # Read each line in the file, readlines() returns a list of lines
    content = file.readlines()
    # Combine the lines in the list into a string
    content = "".join(content)
    bs_content = bs(content, "lxml")


print("UwU")

