from sklearn.ensemble import GradientBoostingClassifier
import evaluation
import time

def GB_classifier(X_train, y_train, X_test, y_test, state):
    a = time.time()
    gb = GradientBoostingClassifier().fit(X_train, y_train)
    b = time.time()
    train_time = b - a

    c = time.time()
    y_pred = gb.predict(X_test)
    d = time.time()
    predict_time = d - c

    gb_probs = gb.predict_proba(X_test)
    gb_probs = gb_probs[:, 1]
    return evaluation.evaluation(y_pred, y_test, state, gb_probs, X_test)
