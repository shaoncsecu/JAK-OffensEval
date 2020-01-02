import SVM_EN_Task_A
import Run_Existing_Model

from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
from gensim.models import Doc2Vec
from sklearn import utils
from gensim.models.doc2vec import TaggedDocument
from sklearn import svm
import multiprocessing
import os

def vec_for_learning(model, tagged_docs):

    # In case no augmented data is supplied
    if len(tagged_docs):
        targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in tagged_docs])
        return targets, regressors

    return [], []


def make_dbow_model(tagged_data):

    print('Training the DBOW Doc2Vec Model...')
    cores = multiprocessing.cpu_count()

    # Build a vocabulary
    # model_dbow = Doc2Vec(dm=0, vector_size=300, negative=5, hs=0, min_count=2, sample = 0, workers=cores)
    model_dbow = Doc2Vec(dm=0, vector_size=512, negative=5, min_count=1, alpha=0.105, min_alpha=0.105, workers=cores)
    model_dbow.build_vocab([x for x in tqdm(tagged_data)])

    # Training a Doc2Vec model is rather straight forward in Gensim, we initialize the model and train for 50 epochs
    for epoch in range(50):
        model_dbow.train(utils.shuffle([x for x in tqdm(tagged_data)]), total_examples=len(tagged_data), epochs=1)
        model_dbow.alpha -= 0.002
        model_dbow.min_alpha = model_dbow.alpha

    return model_dbow


def make_dm_model(tagged_data):

    print('Training the DM Doc2Vec Model...')
    cores = multiprocessing.cpu_count()

    model_dmm = Doc2Vec(dm=1, dm_mean=1, vector_size=512, window=10, negative=5, min_count=1, alpha=0.105, min_alpha=0.105, workers=cores)
    model_dmm.build_vocab([x for x in tqdm(tagged_data)])

    # Training a Doc2Vec model is rather straight forward in Gensim, we initialize the model and train for 50 epochs
    for epoch in range(50):
        model_dmm.train(utils.shuffle([x for x in tqdm(tagged_data)]), total_examples=len(tagged_data), epochs=1)
        model_dmm.alpha -= 0.002
        model_dmm.min_alpha = model_dmm.alpha

    return model_dmm

# This function makes the dataset compitable to gensim
def produce_tagged_doc(trainDoc, trainClass, augmentDoc, augmentClass, testDoc, testClass):

    train_tagged = [TaggedDocument(words=doc, tags=[lbl]) for doc, lbl in zip(trainDoc, trainClass)]
    # test_tagged = [TaggedDocument(words=doc, tags=[lbl]) for doc, lbl in zip(testDoc, testClass)]

    augment_tagged = [TaggedDocument(words=doc, tags=[lbl]) for doc, lbl in zip(augmentDoc, augmentClass)]

    # using fake test labels so that we can train the embedding with test data as well
    test_tagged = [TaggedDocument(words=doc, tags=['test_{}', format(i)]) for i, doc in enumerate(testDoc)]
    # print(train_tagged[0])

    return train_tagged, augment_tagged, test_tagged


def appy_svm(trainDoc, trainClass, testDoc, testClass):

    print('\nTraining the SVM Classifier...!')
    # svm_classifier = svm.SVC(kernel='linear', C=8.5)
    svm_classifier = svm.SVC(kernel='sigmoid', gamma='scale')
    svm_classifier.fit(trainDoc, trainClass)

    # Testing is done in  SVM_EN_Task_A.test_model() function
    # testGuess = svm_classifier.predict(testDoc)
    # print('Testing accuracy %s' % accuracy_score(testClass, testGuess))
    # print('Testing F1 score: {}'.format(f1_score(testClass, testGuess, average='macro')))

    title = 'Non-Linear SVM with Sigmoid Kernel (Gamma = 0.3)'
    SVM_EN_Task_A.test_model(svm_classifier, trainDoc, trainClass, testDoc, testClass, title)

    Run_Existing_Model.save_model(svm_classifier, name='model_EN_A.sav')

    return svm_classifier

def produce_doc2vec(trainDoc, trainClass, augmentDoc, augmentClass, testDoc, testClass):

    train_tagged, augment_tagged, test_tagged = produce_tagged_doc(trainDoc, trainClass, augmentDoc, augmentClass, testDoc, testClass)

    # check if model file already exists
    if not os.path.isfile('model_dbow_en.doc2vec') or not os.path.isfile('model_dm_en.doc2vec'):
        print("\nInitializing English Doc2Vec Model For The First Time...")

        # using training, augmented and test data to make the DBOW embedding vector
        model_dbow = make_dbow_model(train_tagged + augment_tagged + test_tagged)
        model_dbow.save('model_dbow_en.doc2vec')

        # Distributed Memory (DM) with Averaging
        model_dm = make_dm_model(train_tagged + augment_tagged + test_tagged)
        model_dm.save('model_dm_en.doc2vec')

    else:
        print("\nUsing existing English Doc2Vec Model")
        model_dbow = Doc2Vec.load('model_dbow_en.doc2vec')
        model_dm = Doc2Vec.load('model_dm_en.doc2vec')

    model_dbow.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
    model_dm.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)

    
    print('Running Test From The Doc2Vec Class...')
    # concatenate the two models to make an ensemble
    # new_model = ConcatenatedDoc2Vec([model_dbow, model_dm])

    # fake testClass is the test labels that we used when making taggedTestData in doc2vec model
    trainClass, trainDoc = vec_for_learning(model_dbow, train_tagged)
    augmentClass, augmentDoc = vec_for_learning(model_dbow, augment_tagged)
    fake_testClass, testDoc = vec_for_learning(model_dbow, test_tagged)

    # Applying the SVM - training on both trainDoc + augmentDoc
    if len(augmentDoc):
        classifier = appy_svm(trainDoc + augmentDoc, trainClass + augmentClass, testDoc, testClass)
    else:
        classifier = appy_svm(trainDoc, trainClass, testDoc, testClass)

    # print(trainDoc[0])
    # print('')
    # print(trainClass[0])

    return classifier, trainDoc, trainClass, augmentDoc, augmentClass, testDoc, testClass

