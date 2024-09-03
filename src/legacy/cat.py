from catboost import CatBoostClassifier
import src.legacy.evaluation as evaluation
import time

def CAT_classifier(X_train, y_train, X_test, y_test, state):
    a = time.time()
    cat = CatBoostClassifier(verbose=0).fit(X_train, y_train)
    b = time.time()
    train_time = b - a

    c = time.time()
    y_pred = cat.predict(X_test)
    d = time.time()
    predict_time = d - c

    cat_probs = cat.predict_proba(X_test)
    cat_probs = cat_probs[:, 1]
    return evaluation.evaluation(y_pred, y_test, state, cat_probs, X_test)
