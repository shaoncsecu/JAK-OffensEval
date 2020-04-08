import logging
import numpy as np
import argparse
import sys
from time import time
import matplotlib.pyplot as plt
import pickle
from collections import Counter
import emoji
import sklearn.datasets
from scipy.sparse import hstack
import string
import pandas as pd
import csv
from sklearn.linear_model import SGDClassifier, LogisticRegression, LogisticRegressionCV
from sklearn.feature_extraction import stop_words, DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.utils.extmath import density
from sklearn import metrics
from sklearn.datasets.base import Bunch
from sklearn.preprocessing import StandardScaler
import re
import os
from collections import defaultdict
#change model name obviously:
#example python console: python classifier.py --eval --report --confusion_matrix --language=turkish_balanced


# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

# parse commandline arguments
op = argparse.ArgumentParser(description="Run different classifiers")
op.add_argument("--eval",
              action="store_true", dest="do_eval",
              help="Evaluate classifier on dev set.")
op.add_argument("--test",
              action="store_true", dest="do_test",
              help="Use unlabeled test data for predictions.")
op.add_argument("--report",
              action="store_true", dest="print_report",
              help="Print a classification report with precision/recall/f1 score.")
op.add_argument("--plot_useful_features",
              action="store_true", dest="do_plot",
              help="Generate a bar graph of useful features based on their coefficient weights.")
op.add_argument("--confusion_matrix",
              action="store_true", dest="print_cm",
              help="Print a confusion matrix.")
op.add_argument("--top_features",
              action="store_true", dest="print_top",
              help="Print top 100 most discriminative terms for identifying offensive tweets.")
op.add_argument("--grid_search",
              action="store_true", dest="do_search",
              help="Performs a grid search for C values.")
op.add_argument("--language",
              type=str, dest="language",
              help="Specify language vector files.")
op.add_argument("--name",
              type=str, dest="name",
              help="Specify name of classifier for results file.")
op.add_argument("--test_file",
              type=str, dest="og_test",
              help="Original test file data path. Only specify if test case.")
op.add_argument("--model", help="Models: linearsvc, sgd")

opts = op.parse_args()

print(__doc__)
op.print_help()
print()

def is_interactive():
    return not hasattr(sys.modules['__main__'], '__file__')


def pickle_load(file_name):
    return pickle.load(open(file_name, 'rb'))

def run_classifier(clf, eval=False, test=False):
    """
    The function which runs the classifier and outputs performance
    Adapted from: https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html#sphx-glr-auto-examples-text-plot-document-classification-20newsgroups-py
    :param clf: the classifier
    :param eval: bool
    :param test: bool
    :return: None
    """
    print('_' * 80)
    print("Training: ")
    print(clf)
    data_train = pickle_load("vectorized_data/{0}_data_train.p".format(opts.language))
    X_train = pickle_load("vectorized_data/{0}_X_train_vector.p".format(opts.language))
    y_train = pickle_load("vectorized_data/{0}_Y_train_vector.p".format(opts.language))
    target_names = data_train.target_names
    scaler = StandardScaler(with_mean=False)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    if eval:
        data_dev = pickle_load("vectorized_data/{0}_data_dev.p".format(opts.language))
        dev_tweets = pickle_load("processed_data/{0}_dev.p".format(opts.language))
        X_dev = pickle_load("vectorized_data/{0}_X_dev_vector.p".format(opts.language))
        y_dev = pickle_load("vectorized_data/{0}_Y_dev_vector.p".format(opts.language))
        preds_eval = clf.predict(X_dev)
        # output misclassified tweets from dev set so we can look at them and print accuracy
        if not os.path.exists("results/"):
            os.mkdir('results/')
        with open('results/{}_classifier_mislabeled.txt'.format(opts.name), 'w', encoding="utf8") as out_file:
            out_file.write("INDEX ----- PRED ------- TRUE\n")
            incorrect_pred_count = defaultdict(int)
            for i, (t, p) in enumerate(list(zip(y_dev, preds_eval))):
                t = 'NOT' if t == 0 else 'OFF'
                p = 'NOT' if p == 0 else 'OFF'
                if t != p:
                    incorrect_pred_count[p] += 1
                    out_file.write(str(i+1) + ":\t" + p + " ------- " + t + " ------- " + ' '.join(dev_tweets[i][0])+"\n")
            out_file.write("------------------ Pred Count -----------------------\n")
            out_file.write("NOT (false negatives): "+ str(incorrect_pred_count['NOT']))
            out_file.write("\nOFF (false positives): "+ str(incorrect_pred_count['OFF']))
            print("Misclassified tweets written to:", str(out_file))
        score = metrics.f1_score(y_dev, preds_eval)
        print("accuracy:   %0.3f" % score)
    if test:
        X_test = pickle_load("vectorized_data/{0}_X_test_vector.p".format(opts.language))
        preds_test = clf.predict(X_test)
        test_tweets = pd.read_csv("{0}".format(opts.og_test), sep="\t", header=0, encoding="utf8", quoting=csv.QUOTE_NONE)
        test_tweets.columns = ["id", "tweet"]
        test_ids = test_tweets["id"]
        # output test set predictions - 1 with indices so we can see which tweet the classifier assigned which label to and
        # also with no indices so the predictions can be evaluated against the true labels
        if not os.path.exists("results/"):
            os.mkdir('results/')
        with open('results/{}_classifier_test_predictions_with_index.csv'.format(
                opts.name), 'w') as out_file:
            for i, (t,p) in enumerate(list(zip(test_ids, preds_test))):
                if p == 0:
                    out_file.write(str(test_ids[i])+',NOT\n')
                elif p == 1:
                    out_file.write(str(test_ids[i])+',OFF\n')
    t0 = time()
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    if hasattr(clf, 'coef_'):
        print("dimensionality: %d" % clf.coef_.shape[1])
        print("density: %f" % density(clf.coef_))

    if opts.print_report and opts.do_eval:  # print a detailed classification report including P/R/F1
        print("classification report:")
        print(metrics.classification_report(y_dev, preds_eval, labels=[0,1], target_names=target_names))

    if opts.print_cm and opts.do_eval:  # print a confusion matrix
        print("confusion matrix:")
        print(metrics.confusion_matrix(y_dev, preds_eval, labels=[0,1]))
        tp, fp, fn, tn = metrics.confusion_matrix(y_dev, preds_eval, labels=[0,1]).ravel()
        print("True positives:", tp)
        print("False positives:", fp)
        print("True negatives:", tn)
        print("False negatives:", fn)