from xgboost import XGBClassifier
import src.legacy.evaluation as evaluation
import time
from sklearn.metrics import roc_auc_score

def XGB_classifier(X_train, y_train, X_test, y_test, state):
    a = time.time()
    xgb = XGBClassifier(learning_rate = 0.1, max_depth = 3).fit(X_train, y_train)
    b = time.time()
    train_time = (b - a)

    c = time.time()
    y_pred = xgb.predict(X_test)
    d = time.time()
    predict_time = (d - c)

    xgb_probs = xgb.predict_proba(X_test)
    xgb_probs = xgb_probs[:,1]
    return evaluation.evaluation(y_pred, y_test, state, xgb_probs, X_test)

