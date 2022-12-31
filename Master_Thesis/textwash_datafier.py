import pickle
import json
from sklearn.model_selection import train_test_split
import torch
from tqdm import tqdm

#json file is rearragned here in order to form an IOB training file
class JSONDataset(object):
    
    def __init__(self, file_path):
        self.raw_data = json.load(open(file_path))
        self.data = []
        self.labels = []
        self.new_list = []

        
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


    def switch(self, tag):
        label_dict = {"O":0,
                        'B_PERSON_FIRSTNAME':1,
                        'I_PERSON_FIRSTNAME':2,
                        'B_PERSON_LASTNAME':3, 
                        'I_PERSON_LASTNAME':4, 
                        'B_OCCUPATION':5,
                        'I_OCCUPATION':6,
                        'B_LOCATION':7,
                        'I_LOCATION':8,
                        'B_ADDRESS':9,
                        'I_ADDRESS':10,
                        'B_TIME':11,
                        'I_TIME':12,
                        'B_DATE':13,
                        'I_DATE':14,
                        'B_PHONE_NUMBER':15,
                        'I_PHONE_NUMBER':16,
                        'B_EMAIL_ADDRESS':17,
                        'I_EMAIL_ADDRESS':18,
                        'B_ORGANIZATION':19,
                        'I_ORGANIZATION':20,
                        'B_OTHER_IDENTIFYING_ATTRIBUTE':21,
                        'I_OTHER_IDENTIFYING_ATTRIBUTE':22,
                        'B_NUMERICAL':23,
                        'I_NUMERICAL':24,
                        'B_AGE':25,
                        'I_AGE':26}

        ## <--debugging string               
        # if tag != "O":
        #     print("tag: ", tag, label_dict.get(tag, 0))

        # print("switch: ", tag, label_dict.get(tag, 0), "\n")
        return label_dict.get(tag, 0)


    def make_labels_integers(self):

        new_list = []

        for list_of_tokens in tqdm(self.labels):
            temp_list = []
            for j in list_of_tokens:
                temp_list.append(self.switch(j))

            self.new_list.append(temp_list)
        # print(temp_list)
        temp_list = []

        return self



#training file is pickled
ds = JSONDataset('/home/emiel/data/textwash_data.json').load().make_labels_integers()
# ds2 = ds.make_labels_integers()
pickle.dump(ds, open('/home/emiel/data/textwash_data2.pickle', 'wb'))



#pickled file is de-pickled and split into an 80:20 training and test split and splits the training set into another 80:20 split for validation, random seed = 36
class split_file():

    def __init__(self, file_path):
        self.dataset = pickle.load(open(file_path, "rb"))
        self.tokens = self.dataset.data
        self.tags = self.dataset.new_list
        self.token_train = []
        self.token_val = []
        self.token_test = []
        self.tag_train = []
        self.tag_val = []
        self.tag_test = []


    def train_val_test(self):

        token, tag = self.tokens, self.tags
        self.token_train, self.token_test, self.tag_train, self.tag_test = train_test_split(token, tag, test_size=0.2, random_state=36)
        self.token_train, self.token_val, self.tag_train, self.tag_val = train_test_split(self.token_train, self.tag_train, test_size=0.2, random_state=36)

        return self


dataset = split_file('/home/emiel/data/textwash_data2.pickle').train_val_test()
# print(dataset.tag_train[0:2])



class IMDBDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

        # print(self.labels[0])


    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item


    def __len__(self):
        return len(self.labels)

# split_file('/home/emiel/data/textwash_data.pickle').train_val_test()