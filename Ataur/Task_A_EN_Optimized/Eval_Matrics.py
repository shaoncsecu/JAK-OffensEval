import matplotlib.pyplot as plt
import scikitplot as skplt
from sklearn.metrics import *
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.utils.multiclass import unique_labels
import warnings



# This function cross validates the data set into 5 folds
def cross_validation(classifier, trainDoc, trainClass):

    # Showing 10 fold cross validation score cv = no. of folds
    n_fold = 5
    scores = cross_val_score(classifier, trainDoc, trainClass, cv=n_fold)

    print("\n{}-fold Cross Validation on Training + Augmented Data".format(n_fold, scores))
    print("\n{}-fold Cross Validation (Accuracy):\n{}".format(n_fold, scores))
    print("\nAccuracy (Mean - Cross Validation): %f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    scores = cross_val_score(classifier, trainDoc, trainClass, cv=n_fold, scoring='f1_macro')

    print()
    print(n_fold,"-fold Cross Validation (f1-macro):\n", scores)
    print("\nF1-macro (Mean - Cross Validation): %f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# for calculating different scores
def calculate_measures(classifier, testClass, testGuess, title, plot_confusion=True):

    # Compare the accuracy of the output (Yguess) with the class labels of the original test set (Ytest)
    print("Accuracy = "+str(accuracy_score(testClass, testGuess)))
    print("F1-score(macro) = "+str(f1_score(testClass, testGuess, average='macro')))

    # Report on the precision, recall, f1-score of the output (Yguess) with the class labels of the original test set (Ytest)
    print(classification_report(testClass, testGuess, labels=classifier.classes_, target_names=None, sample_weight=None, digits=3))

    # prints the confution matrix in terminal
    print_confusion_matrix(testClass, testGuess, classes=classifier.classes_)

    # Drawing Confusion Matrix
    if plot_confusion:
        plot_confusion_matrix(testClass, testGuess, classes=classifier.classes_, normalize=True, title=title)
        plt.savefig('Confusion_Matrix.png')
        # plt.show()


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


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function Plots the confusion matrix (as image).
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix (Accuracy)'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #     print("Normalized confusion matrix")
    # else:
    #     print('Confusion matrix, without normalization')

    # # if we want to print it in console
    # print(classes)
    # print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


# Draw plots based on different values of some parameter (val)
def draw_plots(val, accu, f1, value_name):
    plt.plot(val, accu, color='red', label='Accuracy')
    plt.plot(val, f1, color='yellow', label='F1-score')

    x_label = "Values of "+value_name
    plt.xlabel(x_label)
    plt.ylabel('Scores')
    plt.legend()

    plt.savefig('Linear_SVM_Plot.png')
    # plt.show()
