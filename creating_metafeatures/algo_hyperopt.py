# pip install hyperopt
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

iris = datasets.load_iris()
X = iris.data  # we only take the first two features.
y = iris.target

ker = ["rbf", "poly","sigmoid"]
gam = ["scale","auto"]
space = {
    "kernel": hp.choice("kernel", ker),
    "gamma": hp.choice("gamma", gam),
}

def hyperparameter_tuning(params):
    clf = SVC(**params)
    acc = cross_val_score(clf, X, y,scoring="f1_macro").mean()
    return {"loss": -acc, "status": STATUS_OK}

trials = Trials()

best = fmin(
    fn=hyperparameter_tuning,
    space = space, 
    algo=tpe.suggest, 
    max_evals=10, 
    trials=trials
)

print("Best: {}".format(best))
x = ker[best.get("kernel")]
y = gam[best.get("gamma")]
print(x,y)