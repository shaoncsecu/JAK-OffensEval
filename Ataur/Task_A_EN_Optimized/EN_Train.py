import Process_Input
import SVM_EN_Task_A
import Manual_Cross_Val
import DocEmbeddings
import EN_Test

import os
import sys


def start_training(train_file_name, test_file_name, re_train=False):
    '''This function checks and trains the model'''

    # check if model file already exists or we supply retrain=True
    if not os.path.isfile('model_EN_A.sav') or re_train:
        print("\nInitializing English Model For The First Time...")
        
        # Reads and Process (tokenize and/or stem) the  Dataset
        # trainDoc, trainClass, testDoc, testClass = Process_Input.process_train_test(
            # train_file = train_file_name, test_file = test_file_name)
        
        # if you want to combine a balanced set out of all data in training folder (pass True)
        trainDoc, trainClass, augmentDoc, augmentClass, testDoc, testClass = Process_Input.process_train_test(
            train_file = train_file_name, test_file = test_file_name, balanced_all=True)

        # This is for doc2vec testing using gensim
        # the data we get are all document vectors of real number instead of tokens
        classifier, trainDoc, trainClass, augmentDoc, augmentClass, testDoc, testClass = \
            DocEmbeddings.produce_doc2vec(trainDoc, trainClass, augmentDoc, augmentClass, testDoc, testClass)

        # # check the best value for gamma
        # SVM_EN_Task_A.run_all_classifiers(trainDoc, trainClass, testDoc, testClass)

        # Calling the SVM Classifiers to make the usual test and cross-validation
        # classifier = SVM_EN_Task_A.run_models(trainDoc + augmentDoc, trainClass + augmentClass, testDoc, testClass)

        # call the manual cross-validation on training set only
        # Manual_Cross_Val.manual_cross_validation(classifier, trainDoc, trainClass, augmentDoc, augmentClass)

    else:
        print("\nEnglish Model File Already Exist...!!! Do you want to re-tain again?")
        ch = input("Provide your Choice [Y/N]: ")

        if ch == 'Y' or ch == 'y':
            print("Training English Model Again...")
            # just call it with re_train=True
            start_training(train_file_name, test_file_name, re_train=True)

        else:
            print('\nProcessing Test on Dev. set ...')
            EN_Test.test_on_devset(model_file = 'model_EN_A.sav', dev_file = test_file_name)


if __name__=='__main__':

    default_train_file = 'data/train/hasoc_en_train_A.tsv'
    default_dev_file = 'data/dev/hasoc_en_gold_A.tsv'

    # if training and testing files are supplied from command line argument
    try:
        start_training(train_file_name = str(sys.argv[1]), test_file_name = sys.argv[2])
    
    except IndexError:
        # uses the default path
        print('\nUsing the default Training/Test file from the data folder...')
        start_training(train_file_name = default_train_file, test_file_name = default_dev_file)
    
    except:
        print('File Not Found...')
        print("Usage: $ python EN_Train.py <training_file> <test_file>")
