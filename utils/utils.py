import ast
import re
import torch
from sklearn.metrics import f1_score
import torch.nn.functional as F

# below method else clause is incorrect. no longer used for gen padding
def pad_tweets(tweets, block_size, pad_token):
    padded_tweets = []
    for tweet in tweets:
        if len(tweet) > block_size:
            padded_tweets.append(tweet[:block_size])
        else:
            padded_tweets.append(F.pad(tweet, (block_size - len(tweet), 0), value=pad_token))
    
    return padded_tweets

def extract_emojis_for_preprocessed(tweets):
    text = " ".join(tweets)
    pattern = re.compile(r':([a-zA-Z_]+):')
    matches = list(set(re.findall(pattern, text)))
    matches = [match.replace("_", "") for match in matches]
    return matches

def extract_emojis_for_original_content(tweets):
    text = " ".join(tweets)
    pattern = re.compile(r':[a-zA-Z_]+:')
    matches = list(set(re.findall(pattern, text)))
    return matches

def remove_emojis(tweets, emojis):
    processed_tweets = tweets
    for emoji in emojis:
        processed_tweets = [tweet.replace(emoji, "").strip() for tweet in processed_tweets]

    processed_tweets = [tweet if len(tweet) > 0 else "." for tweet in processed_tweets]

    return processed_tweets

def clean_original_content(tweets):
    clean_tweets = []
    for tweet in tweets:
        if tweet.startswith("b'") or tweet.startswith('b"'):
            tweet = ast.literal_eval(tweet)
            tweet = tweet.decode()
        tweet = re.sub(r'\s+', ' ', tweet.strip())
        tweet = tweet.lower()
        tweet = re.sub(r'@[\w\d]+', '@user', tweet) # replace @ replies with '@user'
        tweet = re.sub(r'http\S+', 'http', tweet)  # Replace links with 'http'
        clean_tweets.append(tweet)
    return clean_tweets