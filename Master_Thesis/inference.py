from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification, DistilBertTokenizerFast
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from textwash_datafier import JSONDataset
from textwash_datafier import split_file
import textwash_datafier as td
from tqdm import tqdm
import pickle

# dataset = pickle.load(open('/home/emiel/data/textwash_data.pickle', 'rb'))


splitfile = split_file('/home/emiel/data/textwash_data2.pickle').train_val_test()

# print(splitfile.token_train[0]) #<<-- WORKS
# # print(splitfile.token_val[0])
# # print(splitfile.token_test[0])
# print(splitfile.tag_train[0])

TOKENIZER = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")


train_encodings = TOKENIZER(splitfile.token_train, padding=True, truncation=True, is_split_into_words=True, max_length=700)
val_encodings = TOKENIZER(splitfile.token_val, padding=True, truncation=True, is_split_into_words=True, max_length=700)
test_encodings = TOKENIZER(splitfile.token_test, padding=True, truncation=True, is_split_into_words=True, max_length=700)

# print("these are the training encodings \n" , train_encodings[:5] , "\n now it's finished", len(train_encodings)) #<-- WORKS TOO
# print("these are the validation encodings \n" , val_encodings[:5] , "\n now it's finished", len(val_encodings))
# print("these are the test encodings \n" , test_encodings[:5] , "\n now it's finished", len(test_encodings))


train_dataset = td.IMDBDataset(train_encodings, splitfile.tag_train)
val_dataset = td.IMDBDataset(val_encodings, splitfile.tag_val)
test_dataset = td.IMDBDataset(test_encodings, splitfile.tag_test)

# print("test_train:" , train_dataset) #<-- Returns object type for some reason
# print("test_val:" , val_dataset)
# print("test_test: ", test_dataset)
# print(splitfile.tags_val[0:3])
# print("this is the train: ", train_dataset[0]) #<-- works now the labels have been tagged with ints
# print("this is the val: ", val_dataset[0])
# print("this is the test: ", test_dataset[0])



# training_args = TrainingArguments(
#     output_dir='/home/emiel/data/results',          # output directory
#     num_train_epochs=3,                             # total number of training epochs
#     per_device_train_batch_size=16,                 # batch size per device during training
#     per_device_eval_batch_size=64,                  # batch size for evaluation
#     warmup_steps=500,                               # number of warmup steps for learning rate scheduler
#     weight_decay=0.01,                              # strength of weight decay
#     logging_dir='/home/emiel/data/logs',            # directory for storing logs
#     logging_steps=10,
# )

# BERT_MODEL = AutoTokenizer.from_pretrained("bert-base-uncased")

# trainer = Trainer(
#     model=BERT_MODEL,                               # the instantiated 🤗 Transformers model to be trained
#     args=training_args,                             # training arguments, defined above
#     train_dataset=train_dataset,                    # training dataset
#     eval_dataset=val_dataset                        # evaluation dataset
# )

# # # trainer.train()

# import torch
# from torch.utils.data import DataLoader
# from transformers import DistilBertForSequenceClassification, AdamW

# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
# model.to(device)
# model.train()

# train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# optim = AdamW(model.parameters(), lr=5e-5)

# for epoch in range(3):
#     for batch in train_loader:
#         optim.zero_grad()
#         input_ids = batch['input_ids'].to(device)
#         attention_mask = batch['attention_mask'].to(device)
#         labels = batch['labels'].to(device)
#         outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
#         loss = outputs[0]
#         loss.backward()
#         optim.step()

# model.eval()