from classifier_base import *
from sklearn.ensemble import VotingClassifier, StackingClassifier 

#example python console
#test case: python classifier.py --eval --test --test_file=test_data/turkish_test.tsv --report --confusion_matrix --language=turkish --model=linear_svc
#only dev case: python classifier.py --eval --report --confusion_matrix --language=turkish --model=linear_svc

#----------------------
#COMMAND LINE ARGS:
#Run different classifiers

#optional arguments:
#  -h, --help           show this help message and exit
#  --eval               Evaluate classifier on dev set.
#  --test               Use unlabeled test data for predictions.
#  --report             Print a classification report with precision/recall/f1
#                       score.
#  --confusion_matrix   Print a confusion matrix.
#  --language LANGUAGE  Specify language vector files.
#  --test_file OG_TEST  Original test file data path. Only specify if test
#                       case.
#  --model MODEL        Models: linearsvc, sgd, logreg, stacked ensemble
#  --load               Load an existing model: [language]_[model].joblib
#  --repeats REPEATS    How many repeated runs to do.
#  --save_report        Save the classification report.
#  --save_f1            Only save F1 scores.

if opts.model == "linearsvc":
    print('=' * 80)
    print("LinearSVC")
    run_classifier(LinearSVC(C=0.65, penalty="l1", dual=False, tol=0.00001, class_weight='balanced',
                            verbose=True, max_iter=1000), eval=opts.do_eval, test=opts.do_test, repeats=opts.repeats)

elif opts.model == "multinomial":
    print("=" * 80)
    print("Multinomial Naive Bayes")
    run_classifier(MultinomialNB(alpha=0.1), eval=opts.do_eval, test=opts.do_test, repeats=opts.repeats)
    
elif opts.model == "sgd":
    print("="*80)
    print("Linear SVM with SGD")
    run_classifier(SGDClassifier(average=True, learning_rate="constant", eta0=1.0,
                loss="squared_hinge", penalty="l1", max_iter=100, class_weight="balanced"), eval=opts.do_eval, test=opts.do_test, repeats=opts.repeats)

elif opts.model == "logreg":
    print('=' * 80)
    print("Logistic Regression")
    run_classifier(LogisticRegression(solver="liblinear", penalty="l1", C=10, class_weight="balanced", dual=False, verbose=False, random_state=0),
                    eval=opts.do_eval, test=opts.do_test, repeats=opts.repeats)

elif opts.model == "ensemble":
    print('=' * 80)
    print("Ensemble")
    clf1 = LinearSVC(C=0.65, penalty="l1", dual=False, tol=0.00001, class_weight='balanced',
                            verbose=True, max_iter=1000)
    clf2 = SGDClassifier(average=True, learning_rate="constant", eta0=1.0,
                loss="squared_hinge", penalty="l1", max_iter=100, class_weight="balanced")
    clf3 = SGDClassifier(average=True, learning_rate="optimal", alpha=0.0000001,
                loss="squared_hinge", penalty="l1", max_iter=100, class_weight="balanced")
    clf4 = SGDClassifier(average=True, alpha=0.0000001,
                loss="hinge", penalty="l1", max_iter=100, class_weight="balanced")
    clf5 = LinearSVC(C=1, penalty="l1", dual=False, tol=0.00001, class_weight='balanced',
                            verbose=True, max_iter=1000)
    estimators = [("SVC_c065", clf1), ("SGD_constant", clf2), 
                  ("SGD_optimal_sqhinge", clf3), ("SGD_optimal_hinge", clf4), 
                  ("SVC_c1", clf5)]
    eclf = VotingClassifier(estimators=estimators, voting="hard")
    run_classifier(eclf, eval=opts.do_eval, test=opts.do_test, repeats=opts.repeats)

elif opts.model == "stacked_ensemble":
    print('=' * 80)
    print("Stacked Ensemble")
    clf1 = LinearSVC(C=0.65, penalty="l1", dual=False, tol=0.00001, class_weight='balanced',
                            verbose=True, max_iter=1000)
    clf2 = SGDClassifier(average=True, learning_rate="constant", eta0=1.0,
                loss="squared_hinge", penalty="l1", max_iter=100, class_weight="balanced")
    clf3 = SGDClassifier(average=True, learning_rate="optimal", alpha=0.0000001,
                loss="squared_hinge", penalty="l1", max_iter=100, class_weight="balanced")
    clf4 = LogisticRegression(penalty="l1", C=10, class_weight="balanced", solver="liblinear",
                dual=False, random_state=0, max_iter=100)
    
    estimators = [("SVC_c065", clf1), 
                  ("SGD_constant", clf2), 
                  ("SGD_optimal_sqhinge", clf3), 
                  ("LogReg", clf4)]
    final_clf = LinearSVC(C=1, penalty="l1", dual=False, tol=0.00001, class_weight='balanced',
                        verbose=True, max_iter=1000)
    eclf = StackingClassifier(estimators=estimators, final_estimator=final_clf, verbose=True, cv=3)
    run_classifier(eclf, eval=opts.do_eval, test=opts.do_test, repeats=opts.repeats)
        
