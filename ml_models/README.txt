-- Language Processing with Neural Networks Project --

Machine learning models by Johannes

*********************************************************************************************************
Getting results from the machine learning models is a 3-step process:
Preprocessing --> Vectorization --> Model training

All scripts must be in the same folder.
---------------------------------------------------------------------------------------------------------

1. Preprocessing
preprocessing.py

Preprocess datasets

optional arguments:
  -h, --help            show this help message and exit
  --test                Specify if test data input is given.
  -i INPUT, --in_file INPUT
  -o OUTPUT, --out_file OUTPUT

Needs to be called on each dataset: training / dev / test.

Expects tab separated files of the following formats:
Tweet ID \t tweet \t Label  (OR)
Tweet ID \t tweet \t confidence value \t std (OR)
Label \t Tweet (OR in test case)
Tweet ID \t tweet

Example: 
Training: python preprocessing.py --in_file=OffenseEval2020Data/arabic_train.tsv --out_file=arabic_train
Dev.: python preprocessing.py --in_file=OffenseEval2020Data/arabic_dev.tsv --out_file=arabic_dev
Test: python preprocessing.py --in_file=OffenseEval2020Data/arabic_test.tsv --out_file=arabic_test

It is important to decide on a naming convention here and to keep it consistent throughout the pipeline.

---------------------------------------------------------------------------------------------------------

2. Vectorization
vectorizer.py (this is the script to call)
vectorizer_extras.py

optional arguments:
  -h, --help            show this help message and exit
  --train               Specify if train data included
  --test                Specify if test data included
  --dev                 Specify if dev data included
  --train_file TRAIN_INPUT
                        Input file name.
  --dev_file DEV_INPUT  Dev input file name
  --test_file TEST_INPUT
                        Test input file name
  --out_file OUT_FILE   Name format for out file

Should ideally be called with all three preprocessed datasets present.

Example: python vectorizer.py --train --dev --test --train_file=processed_data/arabic_train.p --dev_file=processed_data/arabic_dev.p --test_file=processed_data/arabic_test.p --out_file=arabic

Again, the name format for the output file needs to match the language given in the preprocessing step, in this case "arabic".

---------------------------------------------------------------------------------------------------------

3. Using the models
classifier.py (this is the script to call)
classifier_base.py 

Run different classifiers

optional arguments:
  -h, --help           show this help message and exit
  --eval               Evaluate classifier on dev set.
  --test               Use unlabeled test data for predictions.
  --test_file OG_TEST  Original test file data path. Only specify if test case.
  --report             Print a classification report with precision/recall/f1
                       score.
  --confusion_matrix   Print a confusion matrix.
  --language LANGUAGE  Specify language vector files.
  --model MODEL        Models: linearsvc, sgd, logreg, stacked ensemble
  --load               Load an existing model: [language]_[model].joblib
  --repeats REPEATS    How many repeated runs to do.
  --save_report        Save the classification report.
  --save_f1            Only save F1 scores.

To add different classifiers, simply add a code block in the classifier.py script similar to:
if opts.model == "linearsvc":
    print('=' * 80)
    print("LinearSVC")
    run_classifier(LinearSVC(C=0.65, penalty="l1", dual=False, tol=0.00001, class_weight='balanced',
                            verbose=True, max_iter=1000), eval=opts.do_eval, test=opts.do_test, repeats=opts.repeats)
where LinearSVC(...) is substituted by your chosen model.

To run you need vectorized data files output from the vectorizer script. Need to have universal naming scheme: 
language_train.p --> language_X_train_vector.p / language_Y_train_vector.p

Example: 
 arabic_train.p --> arabic_X_train_vector.p / arabic_Y_train_vector.p
 arabic_dev.p --> arabic_X_dev_vector.p / arabic_Y_dev_vector.p

Examples to run: 
test case: python classifier.py --eval --test --test_file=test_data/arabic_test.tsv --report --confusion_matrix --language=arabic --model=linear_svc (--load)
dev case: python classifier.py --eval --report --confusion_matrix --language=arabic --model=linear_svc (--load)

"--load" can be specified if you have a saved .joblib model in the folder with the classifier script. For the above examples this would be: arabic_linearsvc.joblib
Running the script without "--load" saves the specified {language}_{model}.joblib file automatically.

In dev case, mislabeled tweets for further analysis are saved in "results" folder.
In test case, results are saved in the "results" folder.
F1-scores and classification reports can be saved using "--save_f1" and "--save_report" and are found in the "scores" folder.