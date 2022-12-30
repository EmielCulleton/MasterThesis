from transformers import pipeline
from transformers import AutoTokenizer
from tranfsormers import PreTrainedTokenizer
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from textwash_datafier import JSONDataset
from textwash_datafier import split_file
import textwash_datafier as td
from tqdm import tqdm
import pickle

dataset = pickle.load(open('/home/emiel/data/textwash_data.pickle', 'rb'))


splitfile = split_file('/home/emiel/data/textwash_data.pickle').train_val_test()


TOKENIZER = AutoTokenizer.from_pretrained("bert-base-uncased")


train_encodings = TOKENIZER(splitfile.x_train, padding=True, truncation=True, is_split_into_words=True, max_length=750)
val_encodings = TOKENIZER(splitfile.x_val, padding=True, truncation=True, is_split_into_words=True, max_length=750)
test_encodings = TOKENIZER(splitfile.x_test, padding=True, truncation=True, is_split_into_words=True, max_length=750)


train_dataset = td.TrainingDataset(train_encodings, splitfile.y_train)
val_dataset = td.TrainingDataset(val_encodings, splitfile.y_val)
test_dataset = td.TrainingDataset(test_encodings, splitfile.y_test)



# print(splitfile.y_val[0:3])
# print(val_dataset)


training_args = TrainingArguments(
    output_dir='/home/emiel/data/results',          # output directory
    num_train_epochs=3,                             # total number of training epochs
    per_device_train_batch_size=16,                 # batch size per device during training
    per_device_eval_batch_size=64,                  # batch size for evaluation
    warmup_steps=500,                               # number of warmup steps for learning rate scheduler
    weight_decay=0.01,                              # strength of weight decay
    logging_dir='/home/emiel/data/logs',            # directory for storing logs
    logging_steps=10,
)

MODEL = AutoTokenizer.from_pretrained("bert-base-uncased")

trainer = Trainer(
    model=MODEL,                                    # the instantiated 🤗 Transformers model to be trained
    args=training_args,                             # training arguments, defined above
    train_dataset=train_dataset,                    # training dataset
    eval_dataset=val_dataset                        # evaluation dataset
)

trainer.train()