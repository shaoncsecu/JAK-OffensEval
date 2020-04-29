import logging
import numpy as np
import argparse
import sys
from time import time
import matplotlib.pyplot as plt
import pickle
from collections import Counter
import sklearn.datasets
from scipy.sparse import hstack
import pandas as pd
import csv
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.utils.extmath import density
from sklearn import metrics
import os
from collections import defaultdict
from joblib import dump, load

#Base script for the classifiers. Contains actual run_classifier function.
#Command line arguments can be used to specify language, test or dev case and whether we want to see and save classification information.
#You need vectorized data files output from the vectorizer script. Need to have universal naming scheme: 
#language_train.p --> language_X_train_vector.p / language_Y_train_vector.p

#Example: 
# arabic_train.p --> arabic_X_train_vector.p / arabic_Y_train_vector.p
# arabic_dev.p --> arabic_X_dev_vector.p / arabic_Y_dev_vector.p

#example python console
#test case: python classifier.py --eval --test --test_file=test_data/turkish_test.tsv --report --confusion_matrix --language=turkish --model=linear_svc
#only dev case: python classifier.py --eval --report --confusion_matrix --language=turkish --model=linear_svc

#Run different classifiers

#optional arguments:
#  -h, --help           show this help message and exit
#  --eval               Evaluate classifier on dev set.
#  --test               Use unlabeled test data for predictions.
#  --report             Print a classification report with precision/recall/f1
#                       score.
#  --confusion_matrix   Print a confusion matrix.
#  --language LANGUAGE  Specify language vector files.
#  --test_file OG_TEST  Original test file data path. Only specify if test
#                       case.
#  --model MODEL        Models: linearsvc, sgd, logreg, stacked ensemble
#  --load               Load an existing model: [language]_[model].joblib
#  --repeats REPEATS    How many repeated runs to do.
#  --save_report        Save the classification report.
#  --save_f1            Only save F1 scores.

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
op.add_argument("--confusion_matrix",
              action="store_true", dest="print_cm",
              help="Print a confusion matrix.")
op.add_argument("--language",
              type=str, dest="language",
              help="Specify language vector files.")
op.add_argument("--test_file",
              type=str, dest="og_test",
              help="Original test file data path. Only specify if test case.")
op.add_argument("--model", 
              help="Models: linearsvc, sgd, logreg, stacked ensemble")
op.add_argument("--load", 
              action="store_true", 
              help="Load an existing model: [language]_[model].joblib")
op.add_argument("--repeats", 
              type=int, 
              dest="repeats", 
              help="How many repeated runs to do. Should only specify if interested in comparing model performances. Default = 1")
op.add_argument("--save_report", 
              action="store_true", 
              dest="save_report", 
              help="Save the classification report.")
op.add_argument("--save_f1", 
              action="store_true", 
              dest="save_f1", 
              help="Only save F1 scores.")

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

def run_classifier(clf, eval=False, test=False, repeats=opts.repeats):
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
    if opts.load:
        clf = load("{0}_{1}.joblib".format(opts.language, opts.model))
    if opts.repeats == None:
        repeats = 1
    for i in range(repeats):
        print("Iteration: {0}".format(i))
        t0 = time()
        clf.fit(X_train, y_train)
        train_time = time() - t0
        print("train time: %0.3fs" % train_time)

        if eval:
            dev_tweets = pickle_load("processed_data/{0}_dev.p".format(opts.language))
            X_dev = pickle_load("vectorized_data/{0}_X_dev_vector.p".format(opts.language))
            y_dev = pickle_load("vectorized_data/{0}_Y_dev_vector.p".format(opts.language))
            preds_eval = clf.predict(X_dev)
            # output misclassified tweets from dev set so we can look at them and print F1-score
            if not os.path.exists("results/"):
                os.mkdir('results/')
            with open('results/{0}_{1}_classifier_mislabeled.txt'.format(opts.language, opts.model), 'w', encoding="utf8") as out_file:
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
            print("F1-score:   %0.3f" % score)
        if test:
            X_test = pickle_load("vectorized_data/{0}_X_test_vector.p".format(opts.language))
            preds_test = clf.predict(X_test)
            test_tweets = pd.read_csv("{0}".format(opts.og_test), sep="\t", header=0, encoding="utf8", quoting=csv.QUOTE_NONE)
            test_tweets.columns = ["id", "tweet"]
            test_ids = test_tweets["id"]
            # output test set predictions per OffensEval 2020 format.
            if not os.path.exists("results/"):
                os.mkdir('results/')
            with open('results/{0}_{1}_classifier_test_predictions.csv'.format(opts.language, opts.model), 'w') as out_file:
                for i, (t,p) in enumerate(list(zip(test_ids, preds_test))):
                    if p == 0:
                        out_file.write(str(test_ids[i])+',NOT\n')
                    elif p == 1:
                        out_file.write(str(test_ids[i])+',OFF\n')
        t0 = time()
        test_time = time() - t0
        print("test time:  %0.3fs" % test_time)
        dump(clf, '{0}_{1}.joblib'.format(opts.language, opts.model)) 

        if hasattr(clf, 'coef_'):
            print("dimensionality: %d" % clf.coef_.shape[1])
            print("density: %f" % density(clf.coef_))

        # print a detailed classification report including P/R/F1
        if opts.print_report and opts.do_eval:
            print("classification report:")
            print(metrics.classification_report(y_dev, preds_eval, labels=[0,1], target_names=target_names))
        
        #Save F1-scores
        if opts.save_f1 and opts.do_eval:
            f1 = metrics.f1_score(y_dev, preds_eval, labels=[0, 1], average="macro")
            f1_dict = {"f1": f1}
            df = pd.DataFrame(f1_dict, index=[0])
            if not os.path.exists("scores/"):
                os.mkdir('scores/')
            if not os.path.isfile("scores/{0}_{1}_f1_scores.csv".format(opts.language, opts.model)):
                df.to_csv("scores/{0}_{1}_f1_scores.csv".format(opts.language, opts.model), header="macro f1", sep="\t")
            else:
                df.to_csv("scores/{0}_{1}_f1_scores.csv".format(opts.language, opts.model), mode="a", header=False, sep="\t")
            print("F1 scores saved.")

        #Save classification reports
        if opts.save_report and opts.do_eval:
            report = metrics.classification_report(y_dev, preds_eval, labels=[0,1], output_dict=True)
            if not os.path.exists("scores/"):
                os.mkdir('scores/')
            df = pd.DataFrame(report).transpose()
            if not os.path.isfile("scores/{0}_{1}_classification_report.csv".format(opts.language, opts.model)):
                df.to_csv("scores/{0}_{1}_classification_report.csv".format(opts.language, opts.model))
            else:
                df.to_csv("scores/{0}_{1}_classification_report.csv".format(opts.language, opts.model), mode="a", header=["-","-","-","-"])
            print("Classification report saved.")

        # print a confusion matrix
        if opts.print_cm and opts.do_eval:
            print("confusion matrix:")
            print(metrics.confusion_matrix(y_dev, preds_eval, labels=[0,1]))
            tp, fp, fn, tn = metrics.confusion_matrix(y_dev, preds_eval, labels=[0,1]).ravel()
            print("True positives:", tp)
            print("False positives:", fp)
            print("True negatives:", tn)
            print("False negatives:", fn)