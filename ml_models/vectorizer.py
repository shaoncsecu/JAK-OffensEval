from scipy.sparse import hstack
import argparse
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import sys
import pickle
import sklearn.datasets
import os

#Import additional functions from extra script
from vectorizer_extras import *

#example python console: 
# python vectorizer.py --train --dev --test --train_file=processed_data/arabic_train.p --dev_file=processed_data/arabic_dev.p --test_file=processed_data/arabic_test.p --out_file=arabic

#If you include different features, it is best to decide on a different name format than your chosen baseline naming convention. This helps keep track of different vector file versions.

# parse commandline arguments
op = argparse.ArgumentParser()
op.add_argument("--train", action="store_true", dest="train_case", help="Specify if train data included")
op.add_argument("--test", action="store_true", dest="test_case", help="Specify if test data included")
op.add_argument("--dev", action="store_true", dest="dev_case", help="Specify if dev data included")
op.add_argument("--train_file", type=str, dest="train_input", help="Input file name.")
op.add_argument("--dev_file", type=str, dest="dev_input", help="Dev input file name")
op.add_argument("--test_file", type=str, dest="test_input", help="Test input file name")
op.add_argument("--out_file", type=str, dest="out_file", help="Name format for out file")

opts = op.parse_args()

print(__doc__)
op.print_help()
print()

def is_interactive():
    return not hasattr(sys.modules['__main__'], '__file__')

def pickle_load(file_name):
    """
    Load pickle file of processed data.
    :param file_name: relative path to processed data file.
    :return: loaded pickle file
    """
    return pickle.load(open(file_name, 'rb'))

def pickle_dump1(item, name):
    """
    Dump vector data into pickle file.
    :param item: vector data
    :param name: str (name format for output file)
    :return: None
    """
    if not os.path.exists("vectorized_data/"):
        os.mkdir("vectorized_data/")
    pickle.dump(item, open("vectorized_data/"+name+'.p', "wb"))

def vectorize_data(data, train=False, char=True):
    """
    Extracts features from tweets and combines them into 1 data structure for the classifier
    :param data: data.data bunch returned by `create_data_bunch'
    :param train: bool; whether this is the training data or not
    :return: coordinate sparse matrix of all of the features for every tweet to sent to classifier
    """
    text_untokenized = [untokenize(t) for t in data]
    features = [tweet2features(t) for t in data]

    if train:
        tfidf_word = word_vectorizer.fit_transform(data)
        features = feature_vectorizer.fit_transform(features)
        features = normalizer.fit_transform(features)
        if char:
            tfidf_char = char_vectorizer.fit_transform(text_untokenized)
    else:
        tfidf_word = word_vectorizer.transform(data)
        features = feature_vectorizer.transform(features)
        features = normalizer.transform(features)
        if char:
            tfidf_char = char_vectorizer.transform(text_untokenized)
    if not char:
        vectorized_data = hstack([tfidf_word, features])
    else:
        vectorized_data = hstack([tfidf_word, tfidf_char, features])
    return vectorized_data

# 1 - CREATE VECTORIZERS
#Here you can specify the settings of the vectorizers.

#word level n-grams
word_vectorizer = TfidfVectorizer(analyzer="word", tokenizer=dummy, preprocessor=dummy,
                            max_df=0.60, ngram_range=(1, 3), sublinear_tf=True)

#character level n-grams
char_vectorizer = TfidfVectorizer(max_df=0.6, analyzer="char_wb", lowercase=True,
                                  ngram_range=(3, 5), sublinear_tf=True)
feature_vectorizer = DictVectorizer(dtype=float)
normalizer = sklearn.preprocessing.Normalizer()

#If training data provided
if opts.train_case:
    train_tweets = pickle_load(opts.train_input)
    data_train = create_data_bunch(train_tweets)
    pickle_dump1(data_train, "{0}_data_train".format(opts.out_file))
    target_names = data_train.target_names
    # get labels from train
    y_train = data_train.target
    pickle_dump1(y_train, "{0}_Y_train_vector".format(opts.out_file))
    # vectorize training data
    X_train = vectorize_data(data_train.data, train=True, char=True)
    pickle_dump1(X_train, "{0}_X_train_vector".format(opts.out_file))
    print("Train data loaded and vectorized....")

#If dev data provided
if opts.dev_case:
    dev_tweets = pickle_load(opts.dev_input)
    data_dev = create_data_bunch(dev_tweets)
    pickle_dump1(data_dev, "{0}_data_dev".format(opts.out_file))
    y_dev = data_dev.target
    pickle_dump1(y_dev, "{0}_Y_dev_vector".format(opts.out_file))
    X_dev = vectorize_data(data_dev.data, char=True)
    pickle_dump1(X_dev, "{0}_X_dev_vector".format(opts.out_file))
    print("Dev data loaded and vectorized...")

#If test data provided
if opts.test_case:
    test_tweets = pickle_load(opts.test_input)
    data_test = create_data_bunch(test_tweets, test=True)
    pickle_dump1(data_test, "{0}_data_test".format(opts.out_file))
    X_test = vectorize_data(data_test.data, char=True)
    pickle_dump1(X_test, "{0}_X_test_vector".format(opts.out_file))
    print("Test data loaded and vectorized...")