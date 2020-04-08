import emoji
import pickle
import pandas as pd
import os
import functools
import string
import sys
from twokenize import *
from nltk.tokenize.casual import TweetTokenizer
from nltk import word_tokenize
import time
import csv
import argparse

#example python console: 
# python preprocessing.py --in_file=OffenseEval2020Data/arab_train_balanced.txt --out_file=arab_train
# for testing:
# python preprocessing.py --test --in_file=OffenseEval2020Data/arab_test.txt --out_file=arab_test

op = argparse.ArgumentParser(description="Preprocess datasets")
op.add_argument("--test",
              action="store_true", dest="test_case",
              help="Specify if test data input is given.")
op.add_argument("-i","--in_file",type=str, dest="input")
op.add_argument("-o","--out_file",type=str, dest="output")

opts = op.parse_args()

print(__doc__)
op.print_help()
print()

def is_interactive():
    return not hasattr(sys.modules['__main__'], '__file__')


def process_tweet(tweet):
    """
    Substitute emojis with their string representation and segment tweets.
    """
    em_split_emoji = emoji.get_emoji_regexp().split(tweet)
    em_split_whitespace = [substr.split() for substr in em_split_emoji]
    em_split = functools.reduce(operator.concat, em_split_whitespace)
    tweet = emoji.demojize(" ".join(em_split)).translate(str.maketrans('', '', string.punctuation))
    return tweet.lower()

def read_data(data, test=False):
    """
    Constructs a list of the data, consisting of tuples of [(['tokenized', 'tweet'], 'LABEL'), ...]
    :param data: path to .tsv data file
    :return: list
    """
    data_list = []
    data = pd.read_csv(data, sep="\t", header=0, encoding="utf8", quoting=csv.QUOTE_NONE)
    if test:
        data.columns = ["id", "tweet"]
        tweets = data["tweet"]
        tweets = [tokenizeRawTweetText(process_tweet(r)) for r in tweets]
        data_list = [(twt,) for twt in tweets]
    
    else:
        if len(data.columns) == 2:
            data.columns = ["label", "tweet"]
        elif len(data.columns) == 3:
            data.columns = ["id", "tweet", "label"]
        elif len(data.columns) == 4:
            data.columns = ["id","tweet","label","std"]
        tweets = data["tweet"]
        tweets = [tokenizeRawTweetText(process_tweet(r)) for r in tweets]
    
        if len(data.columns) == 4:
            for tweet, label in zip(tweets, list(data['label'])):
                if label >= 0.5:
                    data_list.append((tweet, "OFF"))
                elif label < 0.5:
                    data_list.append((tweet, "NOT"))
        elif len(data.columns) == 2:
            for tweet, label in zip(tweets, list(data['label'])):
                if label == 'OFF':
                    data_list.append((tweet, 'OFF'))
                elif label == 'NOT':
                    data_list.append((tweet, 'NOT'))
        elif len(data.columns) == 3:
            for tweet, label in zip(tweets, list(data['label'])):
                if label == 'OFF' or label == 1:
                    data_list.append((tweet, 'OFF'))
                elif label == 'NOT' or label == 0:
                    data_list.append((tweet, 'NOT'))
    return data_list

def pickle_dump(item, name):
    if not os.path.exists("processed_data/"):
        os.mkdir("processed_data/")
    pickle.dump(item, open("processed_data/"+name+'.p', "wb"))

def extract_data(file_path, name, test=False):
    if test == True:
        pickle_dump(read_data(file_path, test=True), name)
    else:
        pickle_dump(read_data(file_path, test=False), name)

if opts.test_case:
    extract_data(opts.input, opts.output, test=True)
else:
    extract_data(opts.input, opts.output, test=False)