'''
    ** This program checks if we can read the input and if each Text and Class 
    labels can be read via csv reader without error. Also processes the data.
'''

import os
import csv
from os import listdir
from os.path import isfile, join


def read_csv(file_name, separator):
    '''This function reads the csv file given by the file_name parameter'''
    try:
        f = open(file_name, 'r', newline='', encoding='utf-8')
    except IOError:
        print('Cannot open the file <{}>'.format(file_name))
        raise SystemExit

    csv_read = csv.reader(f, delimiter=separator)
    
    return csv_read


def process_data(data):
    '''This function check if we are getting the same number of comments and the right text/labels
     and also separates the tweets and labels into two different lists'''
    
    # comment count is the count from the first col ("hasoc_english_261744")
    # data count is the count from the csv parser (to double check we are parsing correctly)

    # tweets and class labels will be stored here
    tweets = []
    labels = []

    # data_count starts from the first comments_id
    data = list(data)
        
    for line in data:

        # each line in the categorical 'data' is of 3 elements -> id    tweet   label
        id, tweet, label = line

        # making multiple line tweet to a single line
        tweet = tweet.replace('\n', ' ').replace('\r', ' ')
            
        # to get the count from comment id (i.e., hasoc_english_1)
        coment_count = int(id.strip().split('_')[2])

        # if the following 2 validation is right than the parsing is right
        if label not in ['HOF', 'NOT']:
            # if any of the class label is not right
            print('Parsing Error of Class Label at {}'.format(id))
            raise SystemExit
            
        # append to the corresponding lists
        tweets.append(tweet)
        labels.append(label)

    return tweets, labels


def process_unlabelled_data(data):
    '''This function only separates the test data... doesn't consider sequence'''

    # id's and tweets will be stored here
    ids = []
    tweets = []

    # making list is convenient...
    data = list(data)

    # if the file contains a header (text_id	text) remove it
    id = data[0][0]
    if str(id.strip().split('_')[0]) != 'hasoc':
        del data[0]

    for line in data:

        # each line in the test data contains 2 elements -> text_id     text
        id, tweet = line

        # making multiple line tweet to a single line
        tweet = tweet.replace('\n', ' ').replace('\r', ' ')

        # append to the corresponding lists
        ids.append(id)
        tweets.append(tweet)

    return ids, tweets


def make_balanced(tweets_training, labels_training, tweets_augment, labels_augment):
    '''This function adjust augmented data in a way that if added to the original training it will produce a balance of highes samples'''

    new_tweets_augment = []
    new_labels_augment = []

    # calculate the how much extra HOF/NOT is needed 
    take_hof = 0
    take_not = 0
    count_hof_train = labels_training.count('HOF')
    count_not_train = labels_training.count('NOT')
    count_hof_augmn = labels_augment.count('HOF')
    count_not_augmn = labels_augment.count('NOT')

    if (count_hof_train > count_not_train) and (count_hof_augmn > count_not_augmn):
        take_hof = count_not_augmn - (count_hof_train - count_not_train)
        take_not = count_not_augmn

    elif (count_not_train > count_hof_train) and (count_not_augmn > count_hof_augmn):
        take_hof = count_hof_augmn
        take_not = count_hof_augmn - (count_not_train - count_hof_train)
    
    elif (count_hof_train > count_not_train) and (count_hof_augmn < count_not_augmn):
        take_hof = count_hof_augmn
        take_not = count_hof_augmn + (count_hof_train - count_not_train)

    elif (count_hof_train < count_not_train) and (count_hof_augmn > count_not_augmn):
        take_hof = count_not_augmn + (count_not_train - count_hof_train)
        take_not = count_not_augmn


    # add HOF and NOT to the new lists
    hof_count = 0
    for tweet, lbl in zip(tweets_augment, labels_augment):
        if hof_count == take_hof:
            break

        if lbl == 'HOF':
            new_tweets_augment.append(tweet)
            new_labels_augment.append(lbl)
            hof_count += 1

    not_count = 0
    for tweet, lbl in zip(tweets_augment, labels_augment):
        if not_count == take_not:
            break

        if lbl == 'NOT':
            new_tweets_augment.append(tweet)
            new_labels_augment.append(lbl)
            not_count += 1

    return new_tweets_augment, new_labels_augment
    

def write_to_file(data_frame, file_name):
    '''This function writes the rows into a text.file (According to Dana's Requirements)'''

    with open(file_name, 'w', encoding='utf-8') as file:
        file.write('\n'.join(data_frame))


def read_balanced_training(balanced_all):
    '''Combines all the files in training folder and returns a balanced corpus'''

    train_folder = 'data/train/'
    augment_folder = 'data/augment/'

    # get all the files from training and augmented folder
    train_files = [train_folder+f for f in listdir(train_folder) if isfile(join(train_folder, f))]
    augment_files = [augment_folder+f for f in listdir(augment_folder) if isfile(join(augment_folder, f))]

    print('Documents in Only Training folder')
    print(train_files)
    print('Documents in Augmented Training folder')
    print(augment_files)

    tweets_training = []
    labels_training = []

    # reads all the data in all of the files in the training folder
    for file in train_files:
        data_frame = read_csv(file, separator='\t')
        tweets, labels = process_data(data_frame)
        tweets_training.extend(tweets)
        labels_training.extend(labels)

    tweets_augment = []
    labels_augment = []

    # reads all the data in all of the files in the training folder
    for file in augment_files:
        data_frame = read_csv(file, separator='\t')
        tweets, labels = process_data(data_frame)
        tweets_augment.extend(tweets)
        labels_augment.extend(labels)


    # balancing the augmented dataset w.r.t training data
    if balanced_all:
        tweets_augment, labels_augment = make_balanced(tweets_training, labels_training, tweets_augment, labels_augment)

    return tweets_training, labels_training, tweets_augment, labels_augment


