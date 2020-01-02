import Eval_Matrics
import Run_Existing_Model

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.metrics import *

import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from spacy.lemmatizer import Lemmatizer

import warnings
import time
import sys
import numpy as np


# a dummy function that just returns its input
def identity(x):
    return x

# combines the tokens of a doc in order to use it as char-gram feature
def join_tokens(doc):
    return ''.join(doc)

# NLTK POS Tagger
def tokenize_pos(tokens):
    return [token+"_POS-"+tag for token, tag in nltk.pos_tag(tokens)]


# we are using NLTK stemmer to stem multiple words into root
def apply_word_stemmer(doc):
    '''use this function if you are using word n-grams'''

    # use either Porter or SnowBall Stemmer
    # stemmer = PorterStemmer()
    stemmer = SnowballStemmer(language='english')

    # if you use word n-grams return this:
    return [stemmer.stem(word) for word in doc]


# we are using NLTK stemmer to stem multiple words into root
def apply_char_stemmer(doc):
    '''use this function if you are using character n-grams'''
    
    # use either Porter or SnowBall Stemmer
    # stemmer = PorterStemmer()
    stemmer = SnowballStemmer(language='english')

    # char n-gram requires strings (sentence) instead of tokens (words)
    return ''.join([stemmer.stem(word) for word in doc])


# Using NLTK?WordNet lemmatizer
class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]


# Using Spacy lemmatizer
class SpacyLemmatizer(object):
    def __init__(self):
        self.spcyL = Lemmatizer()

    def __call__(self, doc):
        return [self.spcyL.lemmatizer(t) for t in doc]


class LengthFeatures(BaseEstimator, TransformerMixin):

    def fit(self, x, y=None):
        return self

    def _get_features(self, doc):
        return {"words": len(doc),
                "unique_words": len(set(doc)) }

    def transform(self, raw_documents):
        return [ self._get_features(doc) for doc in raw_documents]


# decide on TF-IDF vectorization for feature
# based on the value of tfidf (True/False)
def tf_idf_func(tfidf):
    # TODO - change the values
    # we use a dummy function as tokenizer and preprocessor,
    # since the texts are already preprocessed and tokenized.
    if tfidf:
        vec = TfidfVectorizer(preprocessor = identity, tokenizer = identity, 
                                ngram_range = (1, 3))
        
        # vec = TfidfVectorizer(preprocessor = apply_word_stemmer, tokenizer = identity, ngram_range = (1, 3))

        # vec = TfidfVectorizer(preprocessor = apply_char_stemmer, 
        #                       tokenizer = SpacyLemmatizer, 
        #                       analyzer='char', ngram_range = (1, 7))
    else:
        vec = CountVectorizer(preprocessor = identity, tokenizer = identity)

    return vec


# Modified TF-IDF vectorization for features: Uses pre-processing
# based on the value of tfidf (True/False)
def tf_idf_func_modified(tfidf):

    if tfidf:
        # Using only tfidf vectorizer
        tfidf_vec = TfidfVectorizer(stop_words ='english', preprocessor = identity,
                                    tokenizer = identity, ngram_range=(1, 3))
        
        # vec = TfidfVectorizer(preprocessor = apply_stemmer, tokenizer = SpacyLemmatizer, 
        #                         lowercase = True, analyzer='char', ngram_range = (1, 7))

        return tfidf_vec

    else:
        # Using Length Vectorizer combined with Tf-Idf and Count Vectorizer (**warning it will take so much time for Linear SVM)
        tfidf_vec = TfidfVectorizer(stop_words='english', preprocessor = identity,
                                    tokenizer = identity, ngram_range=(1, 3))

        count_vec = CountVectorizer(analyzer=identity, stop_words='english',
                                   preprocessor = identity, tokenizer = LemmaTokenizer, ngram_range=(2, 2))

        length_vec = Pipeline([
                        ('textstats', LengthFeatures()),
                        ('vec', DictVectorizer())
                    ])

        # Here we are taking all 3 the above vectorizer (you can remove 1 or 2) - Usually tf-idf works best
        vec = FeatureUnion([("tfidf", tfidf_vec), ("count", count_vec), ('textstats', length_vec)])

        return vec


# Using a SVM Linear Kernel
# SVM Classifier: the value of boolean arg - tfIdf (True/False)
def SVM_Linear(trainDoc, trainClass, testDoc, testClass, tfIdf):
    '''This fuction uses the Linear SVM to train and test and returns a model'''

    # decides on TfidfVectorizer(True) or CountVectorizer(False)
    # if you do no want to apply any pre-processor just use tf_idf_func()
    vec = tf_idf_func(tfIdf)

    # TODO - change the parameters
    # combine the vectorizer with the classifier
    classifier = Pipeline( [('vec', vec),
                            ('cls', svm.SVC(kernel='linear', C=8.5))] )

    t0 = time.time()
    # Fit/Train classifier according to trainDoc, trainClass
    # Here trainDoc are the documents from training set and trainClass is the class labels for those documents
    classifier.fit(trainDoc, trainClass)

    train_time = time.time() - t0
    print("\nTraining Time: ", train_time)

    # Test the model using the testset
    title = 'Binary(HOF/NOT) + Linear SVM + {0}'.format("TfidfVectorizer" if(tfIdf) else "CountVectorizer")
    test_model(classifier, trainDoc, trainClass, testDoc, testClass, title)

    return classifier

# SVM classifiers results for different values of C
# Setting C – (for different accuracy and f1-scores)
def SVM_loop(trainDoc, trainClass, testDoc, testClass, tfIdf):

    # decides on TfidfVectorizer(True) or CountVectorizer(False)
    vec = tf_idf_func(tfIdf)

    # change the range according to your need np.arrange(start, stop, step_size)
    c_values = list(np.arange(7.0, 9.1, 0.1))
    accu_scores = []
    f1_macro = []

    title = 'Binary(HOF/NOT) + Linear SVM + {0}'.format("TfidfVectorizer" if(tfIdf) else "CountVectorizer")
    print("\n##### Output of {} \n For different values of C ({}-{}) #####".format(title, c_values[0], c_values[-1]))

    for c in c_values:

        # combine the vectorizer with the classifier
        classifier = Pipeline( [('vec', vec),
                                ('cls', svm.SVC(kernel='linear', C=c))] )

        classifier.fit(trainDoc, trainClass)
        testGuess = classifier.predict(testDoc)

        # accumulate all the values of accuracy and f1 so that we can draw the plot later
        accu_scores.append(accuracy_score(testClass, testGuess))
        f1_macro.append(f1_score(testClass, testGuess, average='macro'))

        print("C=", round(c,1),"   Accuracy=", accu_scores[-1],"     F1(macro)=", f1_macro[-1])

    Eval_Matrics.draw_plots(c_values, accu_scores, f1_macro, value_name='C')


# Using a Non-Linear Kernel
# SVM Classifier: the value of boolean arg - tfIdf (True/False)
def SVM_NonLinear(trainDoc, trainClass, testDoc, testClass, tfIdf):
    '''This fuction uses the Non-Linear SVM to train and test and returns a model'''

    # decides on TfidfVectorizer(True) or CountVectorizer(False)
    # if you do no want to apply any pre-processor just use tf_idf_func()
    vec = tf_idf_func(tfIdf)

    # TODO - change the parameters
    # combine the vectorizer with the classifier
    classifier = Pipeline( [('vec', vec),
                            ('cls', svm.SVC(kernel='rbf', gamma=0.6, C=2.0))] )

    t0 = time.time()
    # Fit/Train classifier according to trainDoc, trainClass
    # Here trainDoc are the documents from training set and trainClass is the class labels for those documents
    classifier.fit(trainDoc, trainClass)

    train_time = time.time() - t0
    print("\nTraining Time: ", train_time)

    # Test the model using the testset
    title = 'Binary(HOF/NOT) + Non-linear SVM + {0}'.format("TfidfVectorizer" if(tfIdf) else "CountVectorizer")
    test_model(classifier, trainDoc, trainClass, testDoc, testClass, title)

    return classifier


# Non-Linear SVM classifiers results for different values of gamma/C
# Setting C/Gamma – (for different accuracy and f1-scores)
def SVM_Nonlinear_loop(trainDoc, trainClass, testDoc, testClass, tfIdf):

    # decides on TfidfVectorizer(True) or CountVectorizer(False)
    vec = tf_idf_func(tfIdf)

    # change the range according to your need np.arrange(start, stop, step_size)
    gamma_values = list(np.arange(0.1, 10.1, 0.1))
    accu_scores = []
    f1_macro = []

    title = 'Binary(HOF/NOT) + Non-Linear SVM + {0}'.format("TfidfVectorizer" if(tfIdf) else "CountVectorizer")
    print("\n##### Output of {} \n For different values of Gamma ({}-{}) [C= default = 1.0] #####".format(title, gamma_values[0], gamma_values[-1]))

    for g in gamma_values:

        # combine the vectorizer with the classifier
        classifier = Pipeline( [('vec', vec),
                                ('cls', svm.SVC(kernel='rbf', gamma=g))] )

        classifier.fit(trainDoc, trainClass)
        testGuess = classifier.predict(testDoc)

        # accumulate all the values of accuracy and f1 so that we can draw the plot later
        accu_scores.append(accuracy_score(testClass, testGuess))
        f1_macro.append(f1_score(testClass, testGuess, average='macro'))

        print("Gamma=", round(g,1),"   Accuracy=", accu_scores[-1],"     F1(macro)=", f1_macro[-1])

    Eval_Matrics.draw_plots(gamma_values, accu_scores, f1_macro, value_name='Gamma')


# Non-Linear SVM classifiers results for different values of gamma/C
# Setting C/Gamma – (for different accuracy and f1-scores)
def SVM_Nonlinear_loop_doc2vec(trainDoc, trainClass, testDoc, testClass):

    # change the range according to your need np.arrange(start, stop, step_size)
    gamma_values = list(np.arange(0.1, 10.1, 0.1))
    accu_scores = []
    f1_macro = []

    title = 'Binary(HOF/NOT) + Non-Linear SVM + {0}'.format("Doc2Vec")
    print("\n##### Output of {} \n For different values of Gamma ({}-{}) [C= default = 1.0] #####".format(title, gamma_values[0], gamma_values[-1]))

    for g in gamma_values:

        # combine the vectorizer with the classifier
        classifier = svm.SVC(kernel='sigmoid', gamma=g)

        classifier.fit(trainDoc, trainClass)
        testGuess = classifier.predict(testDoc)

        # accumulate all the values of accuracy and f1 so that we can draw the plot later
        accu_scores.append(accuracy_score(testClass, testGuess))
        f1_macro.append(f1_score(testClass, testGuess, average='macro'))

        print("Gamma=", round(g,1),"   Accuracy=", accu_scores[-1],"     F1(macro)=", f1_macro[-1])

    Eval_Matrics.draw_plots(gamma_values, accu_scores, f1_macro, value_name='Gamma')


# SVM classifiers results for different values of C
# Setting C – (for different accuracy and f1-scores)
def SVM_linear_loop_doc2vec(trainDoc, trainClass, testDoc, testClass):

    # change the range according to your need np.arrange(start, stop, step_size)
    c_values = list(np.arange(0.0, 5.1, 0.1))
    accu_scores = []
    f1_macro = []

    title = 'Binary(HOF/NOT) + Linear SVM + {0}'.format("Doc2Vec")
    print("\n##### Output of {} \n For different values of C ({}-{}) #####".format(title, c_values[0], c_values[-1]))

    for c in c_values:

        # combine the vectorizer with the classifier
        classifier = svm.SVC(kernel='linear', C=c)

        classifier.fit(trainDoc, trainClass)
        testGuess = classifier.predict(testDoc)

        # accumulate all the values of accuracy and f1 so that we can draw the plot later
        accu_scores.append(accuracy_score(testClass, testGuess))
        f1_macro.append(f1_score(testClass, testGuess, average='macro'))

        print("C=", round(c,1),"   Accuracy=", accu_scores[-1],"     F1(macro)=", f1_macro[-1])

    Eval_Matrics.draw_plots(c_values, accu_scores, f1_macro, value_name='C')

def test_model(classifier, trainDoc, trainClass, testDoc, testClass, title):
    '''Test the classifier/model using the test data.'''

    t1 = time.time()
    # Use the classifier to predict the class for all the documents in the test set testDoc
    # Save those output class labels in testGuess
    testGuess = classifier.predict(testDoc)

    print("\n########### {} ###########".format(title))

    # Call to function(s) to do the jobs ^_^
    Eval_Matrics.calculate_measures(classifier, testClass, testGuess, title)

    # Doing cross validation on the whole Dataset (takes time!!)
    # Eval_Matrics.cross_validation(classifier, trainDoc, trainClass)

    test_time = time.time() - t1
    print("\nTesting Time: ", test_time)


# This function runs all variations of SVM classifiers
def run_all_classifiers(trainDoc, trainClass, testDoc, testClass):

    # Runs the baseline model (uncomment to run SVM as a baseline)
    # print("\n\n Running the Baseline Model - Linear SVM:")
    # basaeline = SVM_Linear(trainDoc, trainClass, testDoc, testClass, tfIdf=True)

    # Test the Linear SVM with Tf-Idf Vectorizer
    # classifier = SVM_Linear(trainDoc, trainClass, testDoc, testClass, tfIdf=True)

    # # Try different values of C in Linear SVM + CountVectorizer and To collect the data for curve
    # SVM_loop(trainDoc, trainClass, testDoc, testClass, tfIdf=False)

    # # Try different values of C in Linear SVM + Tf-IDF and To collect the data for curve
    # SVM_loop(trainDoc, trainClass, testDoc, testClass, tfIdf=True)

    # Test the Non-Linear SVM with Tf-Idf Vectorizer
    # classifier = SVM_NonLinear(trainDoc, trainClass, testDoc, testClass, tfIdf=True)

    # # Try different values of Gamma in Non-Linear SVM and To collect the data for curve (CountVectorizer)
    # SVM_Nonlinear_loop(trainDoc, trainClass, testDoc, testClass, tfIdf=False)

    # # Try different values of Gamma in Non-Linear SVM and To collect the data for curve (TF-IDF)
    # SVM_Nonlinear_loop(trainDoc, trainClass, testDoc, testClass, tfIdf=True)

    # Try different values of Gamma in Non-Linear SVM and To collect the data for curve (Doc2Vec)
    # SVM_Nonlinear_loop_doc2vec(trainDoc, trainClass, testDoc, testClass)

    # Try different values of C in Linear SVM and To collect the data for curve (Doc2Vec)
    SVM_linear_loop_doc2vec(trainDoc, trainClass, testDoc, testClass)
    pass


# This function runs Best Model with Tf-Idf Vectorizers and some Pre-preprocessing
def run_best_model(trainDoc, trainClass, testDoc, testClass):

    # Test the Non-Linear SVM with Tf-Idf Vectorizer
    classifier = SVM_Linear(trainDoc, trainClass, testDoc, testClass, tfIdf=True)
    
    return classifier


# You should call this function to start everything
def run_models(trainDoc, trainClass, testDoc, testClass):

    # # Runs the best model and save it to the working directory
    print("\nRunning the Best Model - Linear SVM:")
    model = run_best_model(trainDoc, trainClass, testDoc, testClass)
    Run_Existing_Model.save_model(model, name='model_EN_A.sav')

    # Run both linear and no-linear classifier with different C/Gamma values
    # c = input("\n\n Do you want to See the Output of all variants of SVM classifier?:[Y/N]:")
    # if c =='Y' or c == 'y':
    #     # run all the 3 classifiers
    #     run_all_classifiers(trainDoc, trainClass, testDoc, testClass)

    return model
