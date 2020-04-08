from classifier_base import *
from sklearn.ensemble import VotingClassifier, StackingClassifier 
from sklearn.model_selection import cross_val_score
from sklearn import svm

#example python console: python classifier.py --eval --report --confusion_matrix --language=turkish --model=linearsvc (--name=turkish_linear)

if opts.model == "linearsvc":
    print('=' * 80)
    print("LinearSVC")
    run_classifier(LinearSVC(C=0.65, penalty="l1", dual=False, tol=0.00001, class_weight='balanced',
                            verbose=True, max_iter=1000), eval=opts.do_eval, test=opts.do_test, repeats=1)

    print('=' * 80)
    print("Linear SVM with SGD")
    run_classifier(SGDClassifier(average=True, learning_rate="constant", eta0=1.0,
                loss="squared_hinge", penalty="l1", max_iter=100, class_weight="balanced"), eval=opts.do_eval, test=opts.do_test)

elif opts.model == "logreg":
    print('=' * 80)
    print("Logistic Regression")
    run_classifier(LogisticRegression(solver="liblinear", penalty="l1", C=10, class_weight="balanced", dual=False, verbose=False, random_state=0),
                    eval=opts.do_eval, test=opts.do_test)

elif opts.model == "logregcv":
    print('=' * 80)
    print("Logistic Regression CV")
    run_classifier(LogisticRegressionCV(cv=3, verbose=True, penalty="l2", class_weight="balanced", random_state=0),
                    eval=opts.do_eval, test=opts.do_test)

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
    run_classifier(eclf, eval=opts.do_eval, test=opts.do_test)

elif opts.model == "stacked_ensemble":
    print('=' * 80)
    print("Stacked Ensemble")
    clf1 = LinearSVC(C=0.65, penalty="l1", dual=False, tol=0.00001, class_weight='balanced',
                            verbose=True, max_iter=1000)
    clf2 = SGDClassifier(average=True, learning_rate="constant", eta0=1.0,
                loss="squared_hinge", penalty="l1", max_iter=100, class_weight="balanced")
    clf3 = SGDClassifier(average=True, learning_rate="optimal", alpha=0.0000001,
                loss="squared_hinge", penalty="l1", max_iter=100, class_weight="balanced")
    clf4 = LinearSVC(C=1, penalty="l1", dual=False, tol=0.00001, class_weight='balanced',
                            verbose=True, max_iter=1000)
    clf5 = LogisticRegression(penalty="l1", C=10, class_weight="balanced", solver="liblinear",
                dual=False, random_state=0, max_iter=100)
    
    estimators = [("SVC_c065", clf1), ("SGD_constant", clf2), 
                  ("SGD_optimal_sqhinge", clf3), ("SGD_optimal_hinge", clf4), 
                  ("LogReg", clf5)]
    final_clf = LinearSVC(C=0.65, penalty="l1", dual=False, tol=0.00001, class_weight='balanced',
                        verbose=True, max_iter=1000)
    eclf = StackingClassifier(estimators=estimators, final_estimator=final_clf, verbose=True, cv=3)
    run_classifier(eclf, eval=opts.do_eval, test=opts.do_test)
