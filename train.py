from __future__ import print_function
import os
import time
import json
import argparse

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

from utils.plotting import plot_training

from utils.tokenizers import *
from utils.datasets import *
from utils.params import Params
from utils.utils import clean_original_content
import random

seed = 2024
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

device = "cuda" if torch.cuda.is_available() else "cpu"

def train():
    start_time = time.strftime("%d%m%y_%H%M%S")
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str, help="Pass name of model as defined in hparams.yaml.")
    parser.add_argument("--write_data", required = False, default=False, help="Set to true to write_data.")
    args = parser.parse_args()

    # Parse YAML file which has model parameters.
    params = Params("hparams.yaml", args.model_name)

    if not os.path.exists(params.log_dir): os.makedirs(params.log_dir)
    if not os.path.exists(params.save_dir): os.makedirs(params.save_dir)
    if not os.path.exists("figs"): os.makedirs("figs")

    if args.write_data:
        data = pd.read_csv('data/dataset(clean).csv')
        data = shuffle(data, random_state=seed)
        data = data[:params.num_samples]

        train_n = round(len(data)*0.8)
        val_n = round(len(data)*0.9)

        train_data = data.iloc[:train_n]
        val_data = data.iloc[train_n:val_n]
        test_data = data.iloc[val_n:]

        train_data.to_csv(os.path.join(params.data_dir, "train.csv"), index=False)
        val_data.to_csv(os.path.join(params.data_dir, "val.csv"), index=False)
        test_data.to_csv(os.path.join(params.data_dir, "test.csv"), index=False)
    
    train_data = pd.read_csv('data/train.csv')
    val_data = pd.read_csv('data/val.csv')

    emotion_to_int = {
    "disappointed": 0,
    "happy": 1,
    "angry": 2,
    }
    
    labels_train = torch.tensor(list(train_data.Emotion.map(lambda x: emotion_to_int[x])))
    labels_val = torch.tensor(list(val_data.Emotion.map(lambda x: emotion_to_int[x])))

    tweets_train = list(train_data.OriginalContent)
    tweets_val = list(val_data.OriginalContent)

    tweets_train = clean_original_content(tweets_train)
    tweets_val = clean_original_content(tweets_val)

    use_existing_tokenizer = params.use_existing_tokenizer if hasattr(params, 'use_existing_tokenizer') else False
    use_existing_tokens = params.use_existing_tokens if hasattr(params, 'use_existing_tokens') else False

    # set the tokenizer
    if hasattr(params, 'tokenizer'):
        tokenizer_module = __import__('utils.tokenizers',  fromlist=['object'])
        tokenizer_att = getattr(tokenizer_module, params.tokenizer)
        tokenizer = tokenizer_att(use_existing_tokenizer=use_existing_tokenizer, 
                                  use_existing_tokens=use_existing_tokens)
        print("Building tokenizer...")
        tokenizer.build_tokenizer(tweets_train)
        with open(os.path.join(params.save_dir,"tokenizer_{}".format(args.model_name)), 'wb+') as f:
            pickle.dump(tokenizer, f)
    else:
        tokenizer = FastTextTokenizer(use_existing_tokenizer=params.use_existing_tokenizer, 
                                      use_existing_tokens=params.use_existing_tokens)
        print("Building tokenizer...")
        tokenizer.build_tokenizer(tweets_train, remove_emojis=params.remove_emojis)
        tokenizer.save_model(os.path.join(params.save_dir, "tokenizer_{}".format(args.model_name)))

    vocab_size = tokenizer.get_vocab_size()
    print("Vocab size: {:,}".format(vocab_size))
    
    model_args = {'vocab_size': vocab_size}
    if hasattr(params, 'bidirectional'):
        model_args['bidirectional'] = params.bidirectional
    if hasattr(params, 'block_size'):
        model_args['block_size'] = params.block_size
    models_module = __import__('.'.join(['models', params.model_module]),  fromlist=['object'])
    model = getattr(models_module, params.model_name)(**model_args)

    # train/val tags used to save and load tokenized data so no need to recompute tokens in future runs (BPE expensive!)
    print("Building datasets...")
    train_data = model.get_dataset(tweets_train, labels_train, tokenizer, add_special_tokens=params.add_special_tokens,
                                    remove_emojis=params.remove_emojis, tag="train")
    val_data = model.get_dataset(tweets_val, labels_val, tokenizer, add_special_tokens=params.add_special_tokens,
                                 remove_emojis=params.remove_emojis, tag="val")

    val_accs, val_losses, train_losses, train_accs, lrs = [], [], [], [], []

    epochs = params.num_epochs
    learning_rate = params.initial_lr
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=params.min_lr) 

    model = torch.compile(model)
    model.to(device)
    print("model has : {:,} parameters".format(sum(p.numel() for p in model.parameters())))
    best_loss = np.inf
    for epoch in range(1, epochs+1):
        print(f"======Epoch {epoch}/{epochs}=======")

        train_loss, val_loss, train_accuracy, val_accuracy = model.train_model(train_data, val_data, optimizer, batch_size=params.batch_size)
        train_losses.append(train_loss)
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)
        val_losses.append(val_loss)
        last_lr = round(scheduler.get_last_lr()[0], 8)
        lrs.append(last_lr)

        print(f"Epoch {epoch},: Learning Rate: {last_lr}, Train Loss: {train_loss}, Train Acc: {train_accuracy}, Val Loss: {val_loss}, Val Acc: {val_accuracy}")

        # plot function from sample project
        fig = plot_training(train_losses, train_accs,val_losses, val_accs)
        fig.savefig(os.path.join("figs", "{}_{}_training_vis".format(args.model_name, epoch)))

        # Save the model with lowest val loss
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), os.path.join(params.save_dir,"best_{}".format(args.model_name)))

        scheduler.step()
    end_time = time.strftime("%d%m%y_%H%M%S")
    print("Training complete. Start time: ", start_time, "End time: ", end_time)
    

    if hasattr(params, 'fine_tune_epochs'):
        ft_epochs = params.fine_tune_epochs
        model.load_state_dict(torch.load(os.path.join(params.save_dir,"best_{}".format(args.model_name))))
        model.setFineTune()

        optimizer = torch.optim.Adam(model.parameters(), lr=params.initial_lr_ft, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, ft_epochs, eta_min=params.min_lr_ft)
        
        start_time_ft = time.strftime("%d%m%y_%H%M%S")
        print("Fine tuning the model...")
        for epoch in range(1, ft_epochs+1):
            print(f"======Epoch {epoch}/{ft_epochs}=======")

            train_loss, val_loss, train_accuracy, val_accuracy = model.train_model(train_data, val_data, optimizer, batch_size=params.batch_size)
            
            train_losses.append(train_loss)
            train_accs.append(train_accuracy)
            val_accs.append(val_accuracy)
            val_losses.append(val_loss)
            last_lr = scheduler.get_last_lr()[0]
            lrs.append(last_lr)

            print(f"Fine Tune Epoch {epoch},: Learning Rate: {last_lr}, Train Loss: {train_loss}, Train Acc: {train_accuracy}, Val Loss: {val_loss}, Val Acc: {val_accuracy}")

            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), os.path.join(params.save_dir,"best_{}".format(args.model_name)))
            
            scheduler.step()
        print("Fine tuning complete. Start time: ", start_time_ft, "End time: ", time.strftime("%d%m%y_%H%M%S"))

    end_time = time.strftime("%d%m%y_%H%M%S")
    if params.model_name == "Decoder":
        print("\ngenerating sample tweets...")
        for emotion, i in emotion_to_int.items():
            print("\n=== Generating 3 tweets with emotion:", emotion, "===")
            for i in range(3):
                tweet = model.generate(i, 20)
                print("Tweet: " + tokenizer.decode(tweet))

    # Some log information to keep track of model information. 
    logs ={
        "start_time": start_time,
        "end_time": end_time,
        "model": args.model_name,
        "train_losses": train_losses,
        "train_accs": train_accs,
        "val_losses": val_losses,
        "val_accs": val_accs,
        "initial_lr": params.initial_lr,
        "min_lr": params.min_lr,
        "tokenizer": tokenizer.__class__.__name__,
        "remove_emojis": params.remove_emojis,
        "batch_size": params.batch_size,
        "fine_tune_epochs": params.fine_tune_epochs if hasattr(params, 'fine_tune_epochs') else 0,
        "initial_lr_ft": params.initial_lr_ft if hasattr(params, 'initial_lr_ft') else 0,
        "min_lr_ft": params.min_lr_ft if hasattr(params, 'min_lr_ft') else 0
    }

    with open(os.path.join(params.log_dir,"{}_{}.json".format(args.model_name, start_time)), 'w') as f:
        json.dump(logs, f)

if __name__ == '__main__':
    train()
