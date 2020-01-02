import Eval_Matrics
import Process_Input
# import pickle
from joblib import dump, load
import time
from sklearn.pipeline import Pipeline

def save_model(model, name):
    '''saves the model to the working directory'''
    dump(model, name)

    # filename = 'model_A.pkl'
    # with open(filename, 'wb') as fout:
    #     pickle.dump((vectorizer, classifier), fout)


def load_model(name):
    '''loads the model from the working directory'''
    model = load(name)

    # filename = 'model_A.pkl'
    # with open(filename, 'rb') as fin:
    #     vectorizer, classifier = pickle.load(fin)

    return model


def test_existing_model_on_devset(classifier, testDoc, testClass, title):
    '''Test the existing classifier/model using the Development data.'''

    t1 = time.time()

    # Use the classifier to predict the class for all the documents in the test set testDoc
    # Save those output class labels in testGuess
    testGuess = classifier.predict(testDoc)

    print("\n########### {} ###########".format(title))

    # calculates the Accuracy/Precision/F1 etc and prints the confusion matrix
    # if you want to Graphically see the confusion matrix use - 'plot_confusion=True'
    Eval_Matrics.calculate_measures(classifier, testClass, testGuess, title, plot_confusion=False)

    test_time = time.time() - t1
    print("\nTesting Time: ", test_time)


def test_existing_model_on_testset(classifier, testDoc):
    '''Returns the Predicted Test class on the existing classifier/model using the Test data'''

    t1 = time.time()

    # Use the classifier to predict the class for all the documents in the test set testDoc
    # Save those output class labels in testGuess
    testGuess = classifier.predict(testDoc)

    # finding out the labels
    labels = list(set(testGuess))
    
    print('\nTest Done...!!!')
    Process_Input.distribution(testGuess, title='Predicted Test')

    test_time = time.time() - t1
    print("\nTesting Time: ", test_time)

    return testGuess