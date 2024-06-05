import time
import evaluation
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import numpy as np 
import pandas as pd 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from hyperopt import fmin, tpe, hp, STATUS_OK,Trials
from hyperopt.pyll.base import scope
from sklearn import datasets
import warnings
warnings.filterwarnings("ignore")

def knn_classifier(X_train, y_train,X_test,y_test,state):
#     wei = ["uniform", "distance"]
#     nei = [1,2,3,4,5,6,7,8,9]
#     space = {
#         "n_neighbors": hp.choice("n_neighbors", nei),
#         "weights": hp.choice("weights", wei),
#     }

#     def hyperparameter_tuning(params):
#         clf = KNeighborsClassifier(**params)
#         acc = cross_val_score(clf, X_train, y_train,scoring="f1_macro").mean()
#         return {"loss": -acc, "status": STATUS_OK}

#     trials = Trials()

#     best = fmin(
#         fn=hyperparameter_tuning,
#         space = space, 
#         algo=tpe.suggest, 
#         max_evals=3, 
#         trials=trials
#     )

#     # print("Best: {}".format(best))
#     x = wei[best.get("weights")]
#     y = best.get("n_neighbors")
    # print(x,y)
    a = time.time()
    knn = KNeighborsClassifier().fit(X_train, y_train)
    b = time.time()
    train_time=(b - a)
    c = time.time()
    y_pred = knn.predict(X_test)
    d = time.time()
    predict_time=(d - c)
    lr_probs = knn.predict_proba(X_test)
    # lr_probs = lr_probs[:, 1]
    if lr_probs.shape[1] == 1:
        lr_probs = np.hstack([1 - lr_probs, lr_probs])
    else:
        lr_probs = lr_probs[:, 1]
    return evaluation.evaluation(y_pred, y_test,state,lr_probs,X_test)