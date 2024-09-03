from sklearn.ensemble import AdaBoostClassifier
import src.legacy.evaluation as evaluation
import time

def ADA_classifier(X_train, y_train, X_test, y_test, state):
    a = time.time()
    ada = AdaBoostClassifier().fit(X_train, y_train)
    b = time.time()
    train_time = b - a

    c = time.time()
    y_pred = ada.predict(X_test)
    d = time.time()
    predict_time = d - c

    ada_probs = ada.predict_proba(X_test)
    ada_probs = ada_probs[:, 1]
    return evaluation.evaluation(y_pred, y_test, state, ada_probs, X_test)
