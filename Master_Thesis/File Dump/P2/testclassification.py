from datasets import load_dataset
from huggingface_hub import notebook_login
from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification
import code.Master_Thesis.config as config


file = config.TRAINING_FILE.train_test_split(test_size=0.2, shuffle=True)

#print(file["train"][0])

label_list = file["train"].features[f"tokens"].feature.names

example = file["train"][0]
tokenized_input = config.TOKENIZER(example["spans"], is_split_into_words=True)
tokens = config.TOKENIZER.convert_ids_to_tokens(tokenized_input["tokens"])
tokens


# function to realign the tokens and labels, and truncate sequences to be no longer than DistilBERT’s maximum input length:
def tokenize_and_align_labels(examples):
    tokenized_inputs = config.TOKENIZER(examples["spans"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples[f"spans"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

data_collator = DataCollatorForTokenClassification(tokenizer=config.TOKENIZER)

tokenized_file = file.map(tokenize_and_align_labels, batched=True)
#print(tokenized_input)
#print(tokenized_file)