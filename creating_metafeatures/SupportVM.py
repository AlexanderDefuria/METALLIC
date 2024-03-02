import time
import evaluation
import sklearn
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import numpy as np 
import pandas as pd 
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from hyperopt import fmin, tpe, hp, STATUS_OK,Trials
from hyperopt.pyll.base import scope
from sklearn import datasets
import warnings
warnings.filterwarnings("ignore")

def SVM_classifier(X_train, y_train,X_test,y_test,state):
    # ker = ["rbf", "poly","sigmoid"]
    # gam = ["scale","auto"]
    # space = {
    #     "kernel": hp.choice("kernel", ker),
    #     "gamma": hp.choice("gamma", gam),
    # }

    # def hyperparameter_tuning(params):
    #     clf = SVC(**params)
    #     acc = cross_val_score(clf, X_train, y_train,scoring="f1_macro").mean()
    #     return {"loss": -acc, "status": STATUS_OK}

    # trials = Trials()

    # best = fmin(
    #     fn=hyperparameter_tuning,
    #     space = space, 
    #     algo=tpe.suggest, 
    #     max_evals=3, 
    #     trials=trials
    # )

    # # print("Best: {}".format(best))
    # x = ker[best.get("kernel")]
    # y = gam[best.get("gamma")]
    # print(x,y)
    a = time.time()
    # print(y_train)
    # print("")
    SVM = sklearn.svm.SVC(probability=True).fit(X_train, y_train)
    b = time.time()
    train_time=(b - a)
    c = time.time()
    y_pred = SVM.predict(X_test)
    d = time.time()
    predict_time=(d - c)
    lr_probs = SVM.predict_proba(X_test)
    # lr_probs = lr_probs[:, 1]
    if lr_probs.shape[1] == 1:
        lr_probs = np.hstack([1 - lr_probs, lr_probs])
    else:
        lr_probs = lr_probs[:, 1]
    return evaluation.evaluation(y_pred, y_test,state,lr_probs, X_test)