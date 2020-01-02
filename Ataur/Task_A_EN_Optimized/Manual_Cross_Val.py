from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.metrics import *
import matplotlib.pyplot as plt
import scikitplot as skplt
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.utils.multiclass import unique_labels


# a dummy function that just returns its input
def identity(x):
    return x


# decide on TF-IDF vectorization for feature
def tf_idf_func():
    # we use a dummy function as tokenizer and preprocessor,
    # since the texts are already preprocessed and tokenized.
    vec = TfidfVectorizer(preprocessor = identity, tokenizer = identity, 
                            ngram_range = (1, 3))

    return vec


# Using a SVM Linear Kernel - SVM Classifier
# This function will train on training + augmented corpus
# but only cross validate on training set
def manual_cross_validation(classifier, trainDoc, trainClass, augmentDoc, augmentClass):
    '''This fuction uses the Linear SVM to train and test and returns a model'''

    # # if you do no want to apply any pre-processor just use tf_idf_func()
    # vec = tf_idf_func()

    # # combine the vectorizer with the classifier
    # classifier = Pipeline( [('vec', vec),
    #                         ('cls', svm.SVC(kernel='linear', C=8.5))] )

    # converting the augmented set from tuple to list
    augmentDoc = [augmentDoc[i] for i in range(len(augmentDoc))]
    augmentClass = [augmentClass[i] for i in range(len(augmentClass))]

    acc_total = 0
    f1_total = 0
    fold = 10

    # make k folds
    kf = KFold(n_splits = fold)

    k = 1
    for train_index, test_index in kf.split(trainDoc):
        # print("TRAIN:", train_index, "TEST:", test_index)
        
        # split the training doc/class using the folds
        trainDoc_folds = [trainDoc[i] for i in train_index]
        testDoc_fold = [trainDoc[i] for i in test_index]

        trainClass_folds = [trainClass[i] for i in train_index]
        testClass_fold = [trainClass[i] for i in test_index]

        # trainDoc_folds, testDoc_fold = trainDoc[train_index], trainDoc[test_index]
        # trainClass_folds, testClass_fold = trainClass[train_index], trainClass[test_index]

        # adding the training folds with the augmented data
        new_train_doc = augmentDoc + trainDoc_folds
        new_train_class = augmentClass + trainClass_folds

        # running the model
        classifier.fit(new_train_doc, new_train_class)
        testGuess = classifier.predict(testDoc_fold)

        acc = accuracy_score(testClass_fold, testGuess)
        f1 = f1_score(testClass_fold, testGuess, average='macro')
        
        print('\nFold-{}'.format(k))
        k+=1
        print("Accuracy = {}".format(acc))
        print("F1-score(macro) = {}".format(f1))
        print()
        
        # Report on the precision, recall, f1-score of the output (Yguess) with the class labels of the original test set (Ytest)
        print(classification_report(testClass_fold, testGuess, labels=classifier.classes_, target_names=None, sample_weight=None, digits=3))

        # prints the confution matrix in terminal
        print_confusion_matrix(testClass_fold, testGuess, classes=classifier.classes_)

        # for the final average calculation
        acc_total += acc
        f1_total += f1
    
    print('\n###Final Average Resuls:####')
    print('Accuracy Average= {}'.format(acc_total/fold))
    print('F1-score(macro) Average= {}'.format(f1_total/fold))
    

def print_confusion_matrix(testClass, testGuess, classes):
    '''Prints the confusion Matrix in Terminal'''

    # Showing the Confusion Matrix
    print("Confusion Matrix (class):")
    cm = confusion_matrix(testClass, testGuess, labels=classes)
    print(classes)
    print(cm)
    print()

    # Compute confusion matrix of Accuracy
    print("Normalized confusion matrix (Accuracy)")
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(classes)
    print(cm)
