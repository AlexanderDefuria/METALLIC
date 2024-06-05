from sklearn.naive_bayes import GaussianNB
import numpy as np
import evaluation
import time
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

def Gaussian(X_train, y_train,X_test,y_test,state):
    a = time.time()
    dt = GaussianNB().fit(X_train, y_train)
    b = time.time()
    train_time = (b - a)
    c = time.time()
    y_pred = dt.predict(X_test)
    d = time.time()
    predict_time = (d - c)

    lr_probs = dt.predict_proba(X_test)
    # lr_probs = lr_probs[:, 1]
    if lr_probs.shape[1] == 1:
        lr_probs = np.hstack([1 - lr_probs, lr_probs])
    else:
        lr_probs = lr_probs[:, 1]

    return evaluation.evaluation(y_pred, y_test, state, lr_probs, X_test)