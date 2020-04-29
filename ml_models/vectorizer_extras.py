from sklearn.datasets.base import Bunch
import string

#Here, you can include additional feature functions such as punctuation ratio per tweet etc.
#Make sure to call them in the tweet2features function!
#They will be automatically included in the final vectorized data.

def dummy(doc):
    """
    Dummy function to stand in as tokenizer for TfidfVectorizer since our data has already been tokenized
    Not very elegant, but it works
    :param doc: literally anything
    :return: doc
    """
    return doc

def untokenize(doc):
    """
    Used to join a tokenized tweet back together with white space; used for extracting character ngrams
    :param doc:
    :return: string
    """
    return ' '.join(doc)

def get_avg_word_len(text):
    """
    Gets the average word length in the tweet
    :param text: list of strings (tokenized tweet)
    :return: float - average word length
    """
    words = [len(s.translate(str.maketrans('', '', string.punctuation))) for s in text]
    return sum(words) / len(words)

def create_data_bunch(data, test=False):
    """
    Creates a bunch data frame suitable for use with sklearn
    :param data: list of tuples of pre-processed tweets: [(['tweet_word_1', 'tweet_word_2'...], 'label')...]
    :param test: whether or not the data is test data (label/no label)
    :return: sklearn bunch data frame
    """
    examples = [t[0] for t in data if t[0]]
    target = np.zeros((len(data),), dtype=np.int64)
    if not test:
        for i, tweet in enumerate(data):
            if tweet[1] == 'OFF':
                target[i] = 1
            elif tweet[1] == 'NOT':
                target[i] = 0
    dataset = Bunch(data=examples, target=target, target_names=['NOT', 'OFF'])
    return dataset

def tweet2features(tweet):
    """
    Extracts hand-crafted features from a tweet and stores them in a dictionary to be used with
    sklearn's DictVectorizer
    :param tweet: list of tokens (tweet)
    :return: dictionary of {feature: feature_value}
    """
    features = {
        'len(tweet)': len(tweet),
        'avg_word_length': get_avg_word_len(tweet)
    }
    return features