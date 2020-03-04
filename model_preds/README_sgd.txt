Linear SVM with SGD trained using following parameters:
SGDClassifier(alpha=0.0001, average=True, class_weight='balanced',
       epsilon=0.1, eta0=1.0, fit_intercept=True, l1_ratio=0.15,
       learning_rate='constant', loss='squared_hinge', max_iter=100,
       n_iter=None, n_jobs=1, penalty='l1', power_t=0.5, random_state=None,
       shuffle=True, tol=None, verbose=0, warm_start=False)

Emojis turned to text using: https://github.com/carpedm20/emoji
Tokenized using: twokenize (https://github.com/myleott/ark-twokenize-py)

Features are: 
-uni- , bi- and trigram word grams
-tri, four-, and fivegram character grams
-tweet length
-average word length in tweet