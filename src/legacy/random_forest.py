import time
import src.legacy.evaluation as evaluation
import sklearn
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import numpy as np 
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier 
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from hyperopt import fmin, tpe, hp, STATUS_OK,Trials
from hyperopt.pyll.base import scope
from sklearn import datasets
import warnings
warnings.filterwarnings("ignore")

def rf_classifier(X_train, y_train, X_test, y_test, state):
    # est = [100, 200, 300, 400,500]
    # cri = ["gini", "entropy"]

    # space = {
    #     "n_estimators": hp.choice("n_estimators", est),
    #     "max_depth": hp.quniform("max_depth", 1, 7,1),
    #     "criterion": hp.choice("criterion", cri),
    # }

    # def hyperparameter_tuning(params):
    #     clf = RandomForestClassifier(**params,n_jobs=-1)
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
    # x = cri[best.get("criterion")]
    # y = best.get("max_depth")
    # z = est[best.get("n_estimators")]
    # print(x,y,z)
    a = time.time()
    rf = sklearn.ensemble.RandomForestClassifier().fit(X_train, y_train)
    b = time.time()
    train_time = (b - a)
    c = time.time()
    y_pred = rf.predict(X_test)
    d = time.time()
    predict_time = (d - c)

    lr_probs = rf.predict_proba(X_test)
    # lr_probs = lr_probs[:, 1]
    if lr_probs.shape[1] == 1:
        lr_probs = np.hstack([1 - lr_probs, lr_probs])
    else:
        lr_probs = lr_probs[:, 1]
    return evaluation.evaluation(y_pred, y_test, state,lr_probs,X_test)


