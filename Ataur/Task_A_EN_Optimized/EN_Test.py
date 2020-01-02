import sys
import os
import Process_Input
import Run_Existing_Model


def start_testing(model_file, test_file):
    '''Run the model on the Test Data -- no class label given'''
    
    print("Using existing English Model on Test data...")
            
    # Reads and Process (tokenize and/or stem) only Test Dataset
    tweet_ids, testDoc = Process_Input.process_unlabelled_test(file_name = test_file)

    # load the model from parent directory
    model = Run_Existing_Model.load_model(model_file)

    # Test the model using the testset
    testGuess = Run_Existing_Model.test_existing_model_on_testset(model, testDoc)

    # writing the test output
    write_test_output(tweet_ids, testDoc, predicted_labels = testGuess)


def test_on_devset(model_file, dev_file):
    '''This function is used to run the model on the development set -- similar to training set'''

    print("Running existing English Model on Development data...")
            
    # Reads and Process (tokenize and/or stem) only Dev Dataset
    testDoc, testClass = Process_Input.process_test(file_name = dev_file)

    # load the model from parent directory
    model = Run_Existing_Model.load_model(model_file)

    # Test the model using the testset
    title = 'Binary(HOF/NOT) + Linear SVM + TfidfVectorizer'
    Run_Existing_Model.test_existing_model_on_devset(model, testDoc, testClass, title)


def write_test_output(tweet_ids, testDoc, predicted_labels):
    '''Writes the test output to a tab separated file'''

    filename = 'test_out_en.tsv'

    print('\nWriting the Test output on <{}>'.format(filename))
    # TODO - add the testDoc (tweets) in the output if necessary 
    with open(filename, 'w', encoding='utf-8') as file:
        for id, lbl in zip(tweet_ids, predicted_labels):
            file.write('{}\t{}\n'.format(id, lbl))



if __name__=='__main__':

    default_model_file = 'model_EN_A.sav'
    default_dev_file = 'data/dev/hasoc_en_dev_A.tsv'
    default_test_file = 'data/test/hasoc_en_test.tsv'

    # if training and testing files are supplied from command line argument
    try:
        # test_on_devset(model_file = sys.argv[1], dev_file = sys.argv[2])
        start_testing(model_file = sys.argv[1], test_file = sys.argv[2])

    except IndexError:
        # uses the default path
        print('\nUsing the default Model/Test file from the data folder...')
        # test_on_devset(model_file = default_model_file, dev_file = default_dev_file)
        start_testing(model_file = default_model_file, test_file = default_test_file)
    
    except:
        print('File Not Found...')
        print("Usage: $ python EN_Test.py <model_file> <test_file>")