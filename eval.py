from __future__ import print_function
import os
import pickle
import argparse

import pandas as pd
import torch

from utils.tokenizers import FastTextTokenizer
from utils.params import Params
from utils.utils import clean_original_content, remove_emojis

seed = 2024
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

device = "cuda" if torch.cuda.is_available() else "cpu"

def eval():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str, help="Pass name of model as defined in hparams.yaml.")
    args = parser.parse_args()

    params = Params("hparams.yaml", args.model_name)

    test_data = pd.read_csv('data/test.csv')

    emotion_to_int = {
    "disappointed": 0,
    "happy": 1,
    "angry": 2,
    }

    tweets = clean_original_content(list(test_data.OriginalContent))
    
    labels = torch.tensor(list(test_data.Emotion.map(lambda x: emotion_to_int[x])))
    
    # load the trained tokenizer
    if hasattr(params, 'tokenizer'):
        with open(os.path.join(params.save_dir,"tokenizer_{}".format(args.model_name)), 'rb') as f:
            tokenizer = pickle.load(f)
    else:
        tokenizer = FastTextTokenizer(use_existing_tokenizer=True)
        tokenizer.load_model(os.path.join(params.save_dir,"tokenizer_{}".format(args.model_name)))

    vocab_size = tokenizer.get_vocab_size()

    model_args = {'vocab_size': vocab_size}
    if hasattr(params, 'bidirectional'):
        model_args['bidirectional'] = params.bidirectional
    if hasattr(params, 'block_size'):
        model_args['block_size'] = params.block_size
    models_module = __import__('.'.join(['models', params.model_module]),  fromlist=['object'])
    model = getattr(models_module, params.model_name)(**model_args).to(device)

    model.load_state_dict(torch.load(os.path.join(params.save_dir,"best_{}".format(args.model_name))))
    
    full_test_dataset = model.get_dataset(tweets, labels, tokenizer, add_special_tokens=params.add_special_tokens, 
                                          remove_emojis=params.remove_emojis, tag="test")

    if params.model_name == "Decoder":
        acc, loss = model.eval_model(full_test_dataset)
        print("Accuracy:", round(acc, 3))
    else:
        total_metrics, loss = model.eval_model(full_test_dataset)
        print(total_metrics)
    print("Total Average Loss:", round(loss, 3))

if __name__ == '__main__':
    eval()