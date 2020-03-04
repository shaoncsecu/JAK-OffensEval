Stacked Ensemble using https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html#sklearn.ensemble.StackingClassifier

Models: 1x LinearSVC, 3x LinearSVM with SGD, 1x LogisticRegression.
Final Classifier: LinearSVC

Trained on combined train/dev sets.

Emojis turned to text using: https://github.com/carpedm20/emoji
Tokenized using: twokenize (https://github.com/myleott/ark-twokenize-py)

Features are: 
-uni- , bi- and trigram word grams
-tri, four-, and fivegram character grams
-tweet length
-average word length in tweet