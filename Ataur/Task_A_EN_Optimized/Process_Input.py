import Balance_Data
from spacy.lang.en import English
from nltk.corpus import stopwords
import string
import nltk
from bpemb import BPEmb


def read_stopwords():
    '''This function reads the modified stopword list from root source directory'''
    stop_list = []

    with open('stoplist.txt', 'r', encoding='utf-8') as file:
        for line in file:
            stop_list.append(line.strip())
    
    return stop_list


def tokenize_preprocess_corpus(corpus):
    '''This function uses spacy tokenizer'''

    nlp = English()
    # Create a Tokenizer with the default settings for English
    # including punctuation rules and exceptions
    tokenizer = nlp.Defaults.create_tokenizer(nlp)
    
    # making a set from the Enlish stopwords
    try:
        # modified stopword list from file
        stopWords = set(read_stopwords())
    except:
        # use this if you use stopword list directly from nltk
        nltk.download('stopwords')
        stopWords = set(stopwords.words('english'))
    
    documents = []

    for line in corpus:
        word_tokens = []
        tokens = tokenizer(line.strip())    # line.stip() removes extra newline between characters

        for token in tokens:
            tok_str = str(token)
            # removing stopwords and punctuations
            if tok_str.lower() not in stopWords:
                word_tokens.append(tok_str)  # 'token' is not a sting type but rather spacy.token type
                # print(token)

        # appending unique word tokens only per document
        documents.append(word_tokens)
        # documents.append(list(set(word_tokens)))  # appends the tokens i.e., [[t1_d1, t2_d1...],[t1_d2, t2_d2...]...]

    return documents


# Tokenize and Append the text in documents array.
def tokenize_normal(data):
    '''This is a normal python tokenizer'''
    documents = []

    for line in data:
        tokens = line.strip().split()  # tokenize the lines
        documents.append(tokens)  # appends the tokens i.e., [[t1_d1, t2_d1...],[t1_d2, t2_d2...]...]

    return documents


# Make the sentences into byte-pair 
def make_byte_pair(corpus):
    '''This function implements byte-pair encodings'''

    # the bpe model
    bpemb_en = BPEmb(lang="en")

    # we are using the method to remove the stopwords so that the memory usage gets low
    tokenized_corpus = tokenize_preprocess_corpus(corpus)
    documents = []
    
    for word_tokens in tokenized_corpus:
        sentence = ' '.join(word_tokens)
        documents.append(bpemb_en.encode(sentence))

    return documents


# Show Distribution of Data
def distribution(classLabels, title=""):
    '''Shows the districution of the Class'''
    # making sure it's a list (not a numpy array)
    classLabels = list(classLabels)
    
    labels = list(set(classLabels))
    count_class = [0] * len(labels)

    indx = 0
    for label in labels:
        count_class[indx] = classLabels.count(label)
        indx += 1

    print("\nDistribution of classes in {} Set:".format(title))
    print(labels)
    print(count_class)
    print()


def process_train_test(train_file, test_file, balanced_all=False):
    ''' Reads and Processes both the Training and Testing dataset'''

    # Processes English Training corpus and validate the data
    # TODO - balancing will not work for multi label classification
    print('processing the training set...')
    trainDoc_raw, trainClass, augmentDoc_raw, augmentClass = Balance_Data.read_balanced_training(balanced_all)

    # print('processing the (imbalanced) normal training set...')
    # training_data = Read_Data.read_csv(train_file, separator='\t')
    # trainDoc_raw, trainClass = Read_Data.process_data(training_data)

    # Tokenized the documents/tweets (if we do not want to use BPE)
    # trainDoc = tokenize_preprocess_corpus(trainDoc_raw)
    # augmentDoc = tokenize_preprocess_corpus(augmentDoc_raw)

    # Uses byte-pair encoding (no need for separate tokenization)
    trainDoc = make_byte_pair(trainDoc_raw)
    augmentDoc = make_byte_pair(augmentDoc_raw)
    
    # show the distribution of classes in training and testing set
    distribution(trainClass + augmentClass, title='Training')
    
    # Processes English Test corpus and validate the data
    testDoc, testClass = process_test(test_file)

    return trainDoc, trainClass, augmentDoc, augmentClass, testDoc, testClass


def process_test(file_name):
    ''' Reads and Processes Labelled Testing dataset'''

    # Processes English Test corpus and validate the data
    test_data = Balance_Data.read_csv(file_name, separator='\t')
    testDoc_raw, testClass = Balance_Data.process_data(test_data)

    # Tokenized the documents/tweets (if we do not want to use BPE)
    # testDoc = tokenize_preprocess_corpus(testDoc_raw)

    # Uses byte-pair encoding (no need for separate tokenization)
    testDoc = make_byte_pair(testDoc_raw)

    # show the distribution of classes in testing set
    distribution(testClass, title='Test')

    return testDoc, testClass


def process_unlabelled_test(file_name):
    ''' Reads and Processes unlabelled Testing dataset'''

    # Processes English Test corpus and validate the data
    test_data = Balance_Data.read_csv(file_name, separator='\t')
    tweet_ids, testDoc_raw = Balance_Data.process_unlabelled_data(test_data)

    # Tokenized the documents/tweets (if we do not want to use BPE)
    # testDoc = tokenize_preprocess_corpus(testDoc_raw)

    # Uses byte-pair encoding (no need for separate tokenization)
    testDoc = make_byte_pair(testDoc_raw)

    # show the size of the test set
    print('\nSize of Test Data = {}'.format(len(testDoc_raw)))

    return tweet_ids, testDoc
