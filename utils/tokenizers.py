import os
import pickle
import re
import numpy as np
from transformers import DistilBertTokenizer as bt
import torch
import fasttext
from heapq import nlargest
from multiprocessing.dummy import Pool as ThreadPool
pool = ThreadPool(4)

class TokenizerBase:
    def read_existing(self, path):
        with open(path, "rb") as file:
            data = pickle.load(file)
            return data
    
    def write_existing(self, path, data):
        with open(path, "wb+") as file:
            pickle.dump(data, file)
    
    def find_emojis(self, tweets):
        text = " ".join(tweets)
        pattern = re.compile(r':[a-zA-Z_]+:')
        matches = set(re.findall(pattern, text))
        return matches
    
    def remove_strings_from_tweets(self, tweets, strings):
        processed_tweets = tweets
        for s in strings:
            processed_tweets = [tweet.replace(s, "").strip() for tweet in processed_tweets]

        processed_tweets = [tweet if len(tweet) > 0 else "." for tweet in processed_tweets]
        return processed_tweets

# encodes each character individually using utf-8 codepoints
class CharacterTokenizer(TokenizerBase):
    special_tokens = {b"[BEGIN]" : 256, b"[END]" : 257}
    def __init__(self, use_existing_tokenizer=False, use_existing_tokens=False):
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        self.vocab.update(self.special_tokens.values())
        self.emojis = []
    
    def build_tokenizer(self, tweets):
        self.emojis = list(self.find_emojis(tweets))
        self.emojis.sort()

    def encode(self, text, add_special_tokens=False):
        encoded = list(map(int, text.encode("utf-8")))
        encoded = self.add_special_tokens(encoded) if add_special_tokens else encoded
        return encoded
        
    # saving not implemented - not necessary
    def encode_many(self, tweets, block_size=128, add_special_tokens=False, remove_emojis=False, tag=""):
        if remove_emojis:
            tweets = self.remove_strings_from_tweets(tweets, self.emojis)
        encoded_tweets = [torch.tensor(np.array(self.encode(tweet))) for tweet in tweets]
        return encoded_tweets
    
    def decode(self, idx):
        tokens = b"".join([self.vocab[i] for i in idx])
        return tokens.decode("utf-8", errors="replace")

    def get_vocab_size(self):
        return len(self.vocab)
    
    def add_special_tokens(self, tweet):
        tweet = [self.special_tokens[b"[BEGIN]"]]+ tweet + [self.special_tokens[b"[END]"]]
        return tweet
    
    def get_pad_token(self):
        return 0

# encodes each whole word individually
class WordTokenizer(TokenizerBase):
    saved_tokenizer_file = "saved_tokenizers/word_tokenizer.pkl"
    special_tokens = {"[PAD]": 0, "[UNK]": 1, "[BEGIN]" : 2, "[END]" : 3}
    
    def __init__(self, use_existing_tokenizer=False, use_existing_tokens=False):
        self.use_existing_tokenizer = use_existing_tokenizer
        self.vocab = set()
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.emojis = []
        
        if self.use_existing_tokenizer:
            try:
                pre_trained = self.read_existing(self.saved_tokenizer_file)
                self.vocab = pre_trained.get("vocab")
                self.word_to_idx = pre_trained.get("word_to_idx")
                self.idx_to_word = pre_trained.get("idx_to_word")
                self.emojis = pre_trained.get("emojis")
            except Exception as e:
                print(f"Error reading existing tokenizer at {self.saved_tokenizer_file}")
                raise e

    def build_tokenizer(self, tweets):
        if self.use_existing_tokenizer:
            return
        
        self.emojis = list(self.find_emojis(tweets))
        self.emojis.sort()
        
        for tweet in tweets:
            self.vocab.update(tweet.split())

        self.vocab.difference_update(set(self.special_tokens.keys())) # in case a fill token already exists in the text
        offset = len(self.special_tokens)

        # sort the vocab to ensure reproducibility
        ordered_vocab = sorted(list(self.vocab))

        self.word_to_idx = {word: i+offset for i, word in enumerate(ordered_vocab)}
        self.idx_to_word = {i+offset: word for i, word in enumerate(ordered_vocab)}

        self.add_special_tokens_to_tokenizer()

        save_data = {"vocab": self.vocab, "word_to_idx": self.word_to_idx, "idx_to_word": self.idx_to_word, "emojis": self.emojis}
        self.write_existing(self.saved_tokenizer_file, save_data)
    
    def add_special_tokens_to_tokenizer(self):
        for token, i in self.special_tokens.items():
            self.vocab.add(token)
            self.word_to_idx[token] = i
            self.idx_to_word[i] = token
    
    def encode(self, text, add_special_tokens=False):
        unknown_token = self.special_tokens["[UNK]"]
        encoded = [self.word_to_idx.get(word, unknown_token) for word in text.split()]
        encoded = self.add_special_to_tokens(encoded) if add_special_tokens else encoded
        return encoded
    
    def add_special_to_tokens(self, tokens):
        tokens = [self.special_tokens["[BEGIN]"]] + tokens + [self.special_tokens["[END]"]]
        return tokens
    
    def encode_many(self, tweets, add_special_tokens=False, remove_emojis=False, tag=""):
        if remove_emojis:
            tweets = self.remove_strings_from_tweets(tweets, self.emojis)
        encoded = [torch.tensor(np.array(self.encode(tweet, add_special_tokens))) for tweet in tweets]
        return encoded
    
    def decode(self, idx):
        return "".join([self.idx_to_word.get(i) for i in idx])
    
    def get_vocab_size(self):
        return len(self.vocab)
    
    def get_special_tokens(self):
        return self.special_tokens
    
    def get_pad_token(self):
        return self.special_tokens["[PAD]"]

# for use with Bert model
class BertTokenizer(TokenizerBase):
    def __init__(self, max_length=128, use_existing_tokenizer=False, use_existing_tokens=False):
        self.max_length = max_length
        self.tokenizer = bt.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english', do_lower_case=True)
        self.emojis = []

    def build_tokenizer(self, tweets):
        self.emojis = list(self.find_emojis(tweets))
        self.emojis.sort()

    def encode(self, text, add_special_tokens=False):
        encoded_dict = self.tokenizer.encode_plus(
                                text,                     
                                add_special_tokens = add_special_tokens,
                                max_length = self.max_length,
                                padding = "max_length", # pads up to max length
                                truncation = True,      # if longer than max length, truncate
                                return_tensors = 'pt',     # returns pytorch tensors
                                return_attention_mask = True
                                )
        
        return encoded_dict["input_ids"], encoded_dict["attention_mask"]
    
    def encode_many(self, tweets, add_special_tokens=False, remove_emojis=False, tag=""):
        if remove_emojis:
            tweets = self.remove_strings_from_tweets(tweets, self.emojis)
        encoded_tweets = []
        attention_masks = []
        for tweet in tweets:
            ids, mask = self.encode(tweet, add_special_tokens)
            encoded_tweets.append(ids)
            attention_masks.append(mask)

        encoded_tweets = torch.cat(encoded_tweets, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)

        return encoded_tweets, attention_masks
    
    def get_vocab_size(self):
        return 0

# implements the Byte Pair Encoding algorithm
class CustomBPETokenizer(TokenizerBase):
    saved_tokenizer_file = "saved_tokenizers/bpe_tokenizer.pkl"
    saved_tokens_file_base = "saved_tokenizers/bpe_tokens_"
    special_tokens = {"[BEGIN]": 256, "[END]": 257}

    def __init__(self, use_existing_tokenizer=False, use_existing_tokens=False):
        self.use_existing_tokenizer = use_existing_tokenizer
        self.use_existing_tokens = use_existing_tokens

        if self.use_existing_tokenizer:
            try:
                pre_trained = self.read_existing(self.saved_tokenizer_file)
                self.full_vocab = pre_trained.get("full_vocab")
                self.merges = pre_trained.get("merges")
                self.emoji_mapping = pre_trained.get("emoji_mapping")
                self.emojis = pre_trained.get("emojis")
            except Exception as e:
                print("Error reading existing tokens. Ensure files exist and are not corrupted")
                raise e
        else:
            self.full_vocab = {idx: bytes([idx]) for idx in range(256)} # 255 utf-8 chars
            self.full_vocab.update({256: b"[BEGIN]", 257: b"[END]"})
            self.merges = {}
            self.emoji_mapping = {}
            self.emojis = []
        
        self.max_token = 257
    
    def build_tokenizer(self, tweets, num_merges=10000):
        self.num_merges = num_merges
        if self.use_existing_tokenizer:
            return
        
        self.emojis = list(self.find_emojis(tweets))
        self.emojis.sort()

        text = " ".join(tweets)
        
        # first encode predefined tokens
        print("Tokenizing emojis tokens...")
        self.tokenize_emojis()
        print("Completed tokenizing emojis tokens.")

        # integer encode everything and encode emoji tokens
        print("Encoding emojis...")
        encoded = text.encode("utf-8")
        tokens = list(map(int, encoded))
        tokens = self.encode_emojis(tokens)
        print("Completed encoding emojis.")
        
        remaining_merges = self.num_merges
        curr_vocab_size = self.get_vocab_size()
        print("Creating merges...")

        while remaining_merges > 0:
            pairs = self.get_pair_counts(tokens)
            max_pairs = nlargest(remaining_merges, pairs, key=pairs.get)
            pairs_to_merge = {}
            seen = set()
            for pair in max_pairs:
                if pair[0] in seen or pair[1] in seen:
                    break
                seen.update(pair)
                pairs_to_merge[pair] = curr_vocab_size
                self.full_vocab[curr_vocab_size] = self.full_vocab[pair[0]] + self.full_vocab[pair[1]]
                curr_vocab_size += 1
                remaining_merges -= 1
            tokens = self.multi_merge(tokens, pairs_to_merge)
            self.merges.update(pairs_to_merge)
        
        self.max_token = curr_vocab_size

        save_data = {"full_vocab": self.full_vocab, "merges": self.merges, "emojis": self.emojis, "emoji_mapping": self.emoji_mapping}
        self.write_existing(self.saved_tokenizer_file, save_data)
        print("Tokenization Complete.")
    
    # adds tokens for emojis to the vocab
    def tokenize_emojis(self):
        if len(self.emojis) == 0:
            return
        for i, emoji in enumerate(self.emojis):
            i = i+1
            enc = emoji.encode("utf-8")
            tokens = list(map(int, enc))
            self.emoji_mapping[tuple(tokens)] = self.max_token + i
            self.full_vocab[self.max_token + i] = b""
            for c in tokens:
                self.full_vocab[self.max_token + i] += self.full_vocab[c]
        self.max_token = self.max_token + i

    def encode_emojis(self, tokens):
        emojis = list(self.emoji_mapping.keys())
        emojis.sort()

        colon_char = list(map(int, ":".encode("utf-8")))[0]

        new_tokens = tokens.copy()
        i = 0
        while i < len(new_tokens):
            if new_tokens[i] == colon_char:
                for emoji in emojis:
                    if tuple(new_tokens[i:i+len(emoji)]) == emoji:
                        new_tokens[i:i+len(emoji)] = [self.emoji_mapping[emoji]]
                        break
            i += 1

        return new_tokens
    
    def get_pair_counts(self, tokens):
        pairs = {}
        for i in range(len(tokens)-1):
            pair = tokens[i:i+2]
            pair = tuple(pair)
            if pair not in pairs:
                pairs[pair] = 1
                continue
            pairs[pair] += 1
        return pairs
    
    # function allows for merging multiple pairs at once for performance
    def multi_merge(self, tokens, pairs):
        new_tokens = []
        i = 0
        stop = len(tokens) - 1
        while i < stop:
            # if the pair matches, replace it
            if (tokens[i], tokens[i+1]) in pairs:
                new_tokens.append(pairs[(tokens[i], tokens[i+1])])
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        if i == stop:   # if the last pair was not replaced, append the last token
            new_tokens.append(tokens[-1])
        return new_tokens
    
    # replaces a single pair at a time
    def replace(self, pair, new_token, tokens):
        new_tokens = []
        i = 0
        stop = len(tokens)-1
        while i<stop:
            if (tokens[i], tokens[i+1]) == pair:
                i+=2
                new_tokens.append(new_token)
                continue
            new_tokens.append(tokens[i])
            i+=1
        if i == stop:
            new_tokens.append(tokens[-1])
        return new_tokens
    
    def get_vocab_size(self):
        return len(self.full_vocab)

    def decode(self, ints):
        result = b"".join(self.full_vocab[id] for id in ints)
        result = result.decode("utf-8", errors="replace")
        return result

    def encode(self, text, add_special_tokens=False):
        tokens = list(text.encode("utf-8"))
        tokens = self.add_special(tokens) if add_special_tokens else tokens
        tokens = self.encode_emojis(tokens)
        while len(tokens)>1:
            pairs = self.get_pair_counts(tokens)
            # get the pair with the lowest merge count in merges ie. last to be merged
            pair = min(pairs, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            new_token = self.merges[pair]
            tokens = self.replace(pair, new_token, tokens)
        return tokens
    
    def encode_many(self, tweets, add_special_tokens = False, remove_emojis=False, tag=""):
        saved_tokens_file = self.saved_tokens_file_base + tag + "_" + str(remove_emojis) + ".pkl"
        if self.use_existing_tokens:
            if os.path.isfile(saved_tokens_file):
                encoded_tweets = self.read_existing(saved_tokens_file)
                return encoded_tweets
            else:
                fill = tag + "_" + str(remove_emojis)
                print(f"No saved tokens found with tag '{fill}'. Generating new tokens")
        
        if remove_emojis:
            tweets = self.remove_strings_from_tweets(tweets, self.emojis)
        
        encoded_tweets = pool.map(lambda tweet: torch.tensor(np.array(self.encode(tweet, add_special_tokens))), tweets)
        self.write_existing(saved_tokens_file, encoded_tweets)
        
        return encoded_tweets
    
    def add_special(self, tokens):
        tokens = [self.special_tokens["[BEGIN]"]] + tokens + [self.special_tokens["[END]"]]
        return tokens
    
    def get_pad_token(self):
        return 0

# embeds each word with fastText
# cannot pickle fasttext, model saving partially implemented
class FastTextTokenizer(TokenizerBase):
    unsupervised_file_name = "tmp/fasttext_data_unsupervised.txt"
    saved_tokens_file_base = "saved_tokenizers/fasttext_tokens_"
    saved_model_path = "saved_tokenizers/fasttext_model.bin"
    saved_emojis_path = "saved_tokenizers/fasttext_emojis.pkl"

    def __init__(self, use_existing_tokenizer=False, use_existing_tokens=False, dim=32):
        super().__init__()
        self.dim = dim
        self.use_existing_tokenizer = use_existing_tokenizer
        self.use_existing_tokens = use_existing_tokens
        self.model = None
        self.emojis = []
        if use_existing_tokenizer:
            print("Loading existing tokenizer...")
            self.load_model(self.saved_model_path)
            self.emojis = self.read_existing(self.saved_emojis_path)

    def build_tokenizer(self, x, remove_emojis=False):
        if self.use_existing_tokenizer:
            return
        self.emojis = list(self.find_emojis(x))
        self.emojis.sort()
        self.write_existing(self.saved_emojis_path, self.emojis)
        self.create_unsupervised_file(x)
        self.train_unsupervised_from_file()
        self.remove_unsupervised_file()
        self.save_model(self.saved_model_path)
    
    def create_unsupervised_file(self, x):
        text = "\n".join(x)
        os.makedirs("tmp", exist_ok=True)
        with open(self.unsupervised_file_name, "w+") as f:
            f.write(text)
    
    def train_unsupervised_from_file(self):
        self.model = fasttext.train_unsupervised(self.unsupervised_file_name, dim=self.dim)

    def encode(self, text):
        return [self.model.get_word_vector(word) for word in text.split()]
    
    def encode_many(self, tweets, add_special_tokens=False, remove_emojis=False, tag=""):
        saved_tokens_file = self.saved_tokens_file_base + tag + "_" + str(remove_emojis) + ".pkl"
        if self.use_existing_tokens:
            if os.path.isfile(saved_tokens_file):
                encoded_tweets = self.read_existing(saved_tokens_file)
                return encoded_tweets
            else:
                fill = tag + "_" + str(remove_emojis)
                print(f"No saved tokens found with tag '{fill}'. Generating new tokens")

        if remove_emojis:
            tweets = self.remove_strings_from_tweets(tweets, self.emojis)
        
        encoded = [torch.tensor(np.array(self.encode(tweet))) for tweet in tweets]

        self.write_existing(saved_tokens_file, encoded)

        return encoded
    
    def remove_unsupervised_file(self):
        os.remove(self.unsupervised_file_name)

    def get_dimension(self):
        return self.model.get_dimension()
    
    def save_model(self, filepath):
        self.model.save_model(filepath)
    
    def load_model(self, filepath):
        self.model = fasttext.load_model(filepath)
    
    def get_vocab_size(self):
        return 0