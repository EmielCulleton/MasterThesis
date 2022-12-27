import pandas as pd
import dataset
from datasets import Dataset

#directly importing it as json didn't work might improve this later
textwash_json = pd.read_json('/home/emiel/data/textwash_data.json')
training_file_pandas = pd.DataFrame(textwash_json)
training_data_json = Dataset.from_pandas(training_file_pandas)
#print(training_data_json)
train_test_valid = training_data_json.train_test_split(test_size=0.2, shuffle=True)
