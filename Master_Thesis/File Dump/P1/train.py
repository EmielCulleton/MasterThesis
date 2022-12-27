import pandas as pd
import numpy as np

import joblib
import torch
from tqdm import tqdm, trange

from sklearn import preprocessing
from sklearn import model_selection

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

import config
import code.Master_Thesis.create_dataset as create_dataset
import engine
from model import EntityModel

def process_data(data_path):
    df = config.TRAINING_FILE
    #df = pd.read_csv(data_path, encoding="latin-1") #load file
    df.loc[:, "text"] = df["text"].fillna(method="ffill")

    enc_token = preprocessing.LabelEncoder()
    enc_spans = preprocessing.LabelEncoder()

    df.loc[:, "tokens"] = enc_token.fit_transform(df["tokens"])
    df.loc[:, "spans"] = enc_spans.fit_transform(df["spans"])

    sentences = df.groupby("Sentence #")["Word"].apply(list).values
    token = df.groupby("text")["token"].apply(list).values
    spans = df.groupby("text")["spans"].apply(list).values
    return sentences, token, spans, enc_token, enc_spans

if __name__ == "__main__":
    sentences, token, spans, enc_token, enc_spans = process_data(config.TRAINING_FILE) #load training file
    
    meta_data = {
        "enc_token": enc_token,
        "enc_spans": enc_spans
    }

    joblib.dump(meta_data, "meta.bin")

    num_token = len(list(enc_token.classes_))
    num_spans = len(list(enc_spans.classes_))

    (
        train_sentences,
        test_sentences,
        train_pos,
        test_pos,
        train_tag,
        test_tag
    ) = model_selection.train_test_split(sentences, pos, tag, random_state=36, test_size=0.1)

    train_dataset = create_dataset.EntityDataset(
        texts=train_sentences, pos=train_pos, tags=train_tag
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.TRAIN_BATCH_SIZE, num_workers=4
    )

    valid_dataset = create_dataset.EntityDataset(
        texts=test_sentences, pos=test_pos, tags=test_tag
    )

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=config.VALID_BATCH_SIZE, num_workers=1
    )

    device = torch.device("cuda")
    model = EntityModel(num_tag=num_tag, num_pos=num_pos)
    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.001,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    num_train_steps = int(len(train_sentences) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
    optimizer = AdamW(optimizer_parameters, lr=3e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )

    best_loss = np.inf
    for epoch in range(config.EPOCHS):
        train_loss = engine.train_fn(train_data_loader, model, optimizer, device, scheduler)
        test_loss = engine.eval_fn(valid_data_loader, model, device)
        print(f"Train Loss = {train_loss} Valid Loss = {test_loss}")
        if test_loss < best_loss:
            torch.save(model.state_dict(), config.MODEL_PATH)
            best_loss = test_loss

    print("done")