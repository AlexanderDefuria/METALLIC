import imblearn
import numpy as np
import sklearn
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_fscore_support

"""
This function is used to evaluate the performance of the given model.

It takes in the following parameters:
1. y_pred: The predicted values
2. y_true: The true values
3. state: The state of the classification problem
4. lr_probs: The probability of the predicted values
5. X_test: The test data

It returns the following:
1. f1_score: The f1 score of the model
2. accuracy: The accuracy of the model
3. geometric_mean: The geometric mean of the model
4. precision: The precision of the model
5. recall: The recall of the model
6. roc_auc: The roc auc score of the model
7. pr_auc: The pr auc score of the model
8. balanced_accuracy: The balanced accuracy of the model
9. cwa: The class weighted accuracy of the model

"""


def evaluation(y_pred, y_true, state, lr_probs, X_test):
    au_y_true = y_true
    au_lr = lr_probs
    f1_score = sklearn.metrics.f1_score(y_true, y_pred, average="macro" if state == "multiclass" else "binary")
    accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)
    geometric_mean = imblearn.metrics.geometric_mean_score(y_true, y_pred, average="multiclass" if state == "multiclass" else "binary")  # geometric mean
    precision = sklearn.metrics.precision_score(y_true, y_pred, labels=np.unique(y_pred), average="weighted" if state == "multiclass" else "binary")
    recall = sklearn.metrics.recall_score(y_true, y_pred, average="weighted" if state == "multiclass" else "binary")
    balanced_accuracy = sklearn.metrics.balanced_accuracy_score(y_true, y_pred)

    # TODO THIS IS AN ERROR?????? IT DOES NOT LINE UP WITH FORUMLA IN THE PAPER
    y_true = np.argmax(X_test, axis=1)

    if state == "multiclass":
        roc_auc = -1
    else:
        try:
            roc_auc = roc_auc_score(au_y_true, au_lr)
        except:
            roc_auc = 0.5


    if state == "multiclass":
        pr_auc = -1
    else:
        try:
            precision, recall, _ = precision_recall_curve(au_y_true, au_lr)
            pr_auc = auc(recall, precision)
        except:
            pr_auc = 0.5

    weights = sklearn.utils.class_weight.compute_class_weight(class_weight="balanced", classes=np.unique(y_true), y=y_true)
    prfs = precision_recall_fscore_support(y_true, y_pred, average=None, labels=np.unique(y_true))
    recalls = prfs[1]  # Specificity in Binary Classification
    s = sum(weights)
    new_wei = weights / s
    cwa = sum(new_wei * recalls)
    
    return f1_score, accuracy, geometric_mean, precision, recall, roc_auc, pr_auc, balanced_accuracy, cwa
