import pickle
import json
from sklearn.model_selection import train_test_split
import torch


#json file is rearragned here in order to form an IOB training file
class JSONDataset(object):
    
    def __init__(self, file_path):
        self.raw_data = json.load(open(file_path))
        self.data = []
        self.labels = []
    
    def load(self):
        for json_line in self.raw_data:
            tokens, spans = json_line['tokens'], json_line['spans']
            text, labels = [x['text'] for x in tokens], ['O'] * len(tokens)
            for span in spans:
                labels[span['token_start']] = 'B_' + span['label']
                if span['token_start'] != span['token_end']:
                    labels[span['token_start'] + 1] = 'I_' + span['label']
                    if (span['token_start'] + 1) < span['token_end']:
                        labels[span['token_start'] + 2: span['token_end'] + 1] = \
                        ['I_' + span['label']] * (span['token_end'] - span['token_start'] - 1)
            self.data.append(text) 
            self.labels.append(labels)
        return self


#training file is pickled
ds = JSONDataset('/home/emiel/data/textwash_data.json').load()
pickle.dump(ds, open('/home/emiel/data/textwash_data.pickle', 'wb'))


#pickled file is de-pickled and split into an 80:20 training and test split and splits the training set into another 80:20 split for validation, random seed = 36
class split_file():

    def __init__(self, file_path):
        self.dataset = pickle.load(open(file_path, "rb"))
        self.tokens = self.dataset.data
        self.tags = self.dataset.labels
        self.x_train = []
        self.x_val = []
        self.x_test = []
        self.y_train = []
        self.y_val = []
        self.y_test = []


    def train_val_test(self):

        x, y = self.tokens, self.tags
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.2, random_state=36)
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(self.x_train, self.y_train, test_size=0.2, random_state=36)

        return self


class TrainingDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

split_file('/home/emiel/data/textwash_data.pickle').train_val_test()