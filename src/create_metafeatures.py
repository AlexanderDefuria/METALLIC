from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
from threading import Lock
from typing import Dict, List, Any
import copy
import itertools

from catboost import CatBoostClassifier
from imblearn.base import BaseSampler
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, re
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import (
    auc,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import compute_class_weight
from data_preprocessing import get_datasets, preprocess
from xgboost import XGBClassifier
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import (
    SMOTE,
    ADASYN,
    BorderlineSMOTE,
    SVMSMOTE,
    RandomOverSampler,
)
from imblearn.under_sampling import (
    NearMiss,
    RandomUnderSampler,
    ClusterCentroids,
    TomekLinks,
    EditedNearestNeighbours,
    RepeatedEditedNearestNeighbours,
    AllKNN,
    CondensedNearestNeighbour,
    NeighbourhoodCleaningRule,
    InstanceHardnessThreshold,
)
from imblearn.metrics import geometric_mean_score
from tqdm import tqdm
from skopt import BayesSearchCV
from skopt.space import Categorical, Integer, Real
from sklearn.preprocessing import MinMaxScaler
import builtins


from hyperparameters import HYPERPARAMETERS
import argparse
from calculate_metafeatures import calculate_metafeatures



def print(*args, **kwargs):
    if not QUIET:
        builtins.print(*args, **kwargs)

def get_classifiers(choice: str | None = None) -> Dict[str, object]:
    classifiers = {
        "xgb": XGBClassifier(),
        "knn": KNeighborsClassifier(),
        "dt": DecisionTreeClassifier(),
        "gnb": GaussianNB(),
        "svm": SVC(probability=True),
        "rf": RandomForestClassifier(),
        "ada": AdaBoostClassifier(),
        "cat": CatBoostClassifier(verbose=0),
    }
    return classifiers if choice is None else {choice: classifiers[choice]}


def get_resamplers(choice: str | None = None) -> Dict[str, object]:
    """
    Returns a dictionary of resamplers.
    Returns:
        Dict[str, object]: A dictionary where the keys are the names of the resamplers and the values are the corresponding resampler objects.
    """
    resamplers = {
        "none": None,
        "smote": SMOTE(random_state=42),
        "smoteenn": SMOTEENN(
          random_state=42,
          sampling_strategy="minority",
          smote=SMOTE(random_state=42),
          enn=EditedNearestNeighbours(sampling_strategy="majority", kind_sel="mode"),
        ),
        "randomoversampler": RandomOverSampler(random_state=42),
        "adasyn": ADASYN(random_state=42, sampling_strategy="minority"),
        "borderlinesmote": BorderlineSMOTE(random_state=42),
        "smotesvm": SVMSMOTE(random_state=42),
        "randomundersampler": RandomUnderSampler(random_state=42, sampling_strategy="majority"),
        "clustercentroids": ClusterCentroids(random_state=42),
        "nearmissv1": NearMiss(version=1),
        "nearmissv2": NearMiss(version=2),
        "nearmissv3": NearMiss(version=3),
        "tomeklinks": TomekLinks(),
        "editednn": EditedNearestNeighbours(sampling_strategy="majority", kind_sel="mode"),
        "repeatededitednn": RepeatedEditedNearestNeighbours(sampling_strategy="majority", kind_sel="mode"),
        "allknn": AllKNN(sampling_strategy="majority", kind_sel="mode"),
        "condensednn": CondensedNearestNeighbour(random_state=42, sampling_strategy="majority"),
        "ncl": NeighbourhoodCleaningRule(sampling_strategy="majority", threshold_cleaning=0.2),
        "instancehardness": InstanceHardnessThreshold(random_state=42),
        "smotetomek": SMOTETomek(random_state=42, smote=SMOTE(random_state=42)),
    }
    return resamplers if choice is None else resamplers[choice]


def train_classifiers(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    classifier_name: str,
    resampler_name: str,
    file_name: str,
) -> Dict[str, Any] | None:
    """
    Resample the dataset and train the classifier

    Returns the classifier's f1 score, accuracy, geometric mean, precision, recall, roc auc, pr auc, balanced accuracy, and class weighted accuracy
    """
    np.int = np.int32  # type: ignore
    np.float = np.float64  # type: ignore
    np.bool = np.bool_  # type: ignore
    internal_fold_count = 2
    print(f"Training {classifier_name} with {resampler_name} for {file_name} using {internal_fold_count} CFV")

    # Get the classifier and resampler
    classifier: BaseEstimator = get_classifiers()[classifier_name]  # type: ignore
    resampler: BaseSampler | None = get_resamplers()[resampler_name]  # type: ignore

    # Scale the dataset
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    classes: int = np.concatenate((y_train, y_test)).max() + 1

    if classifier_name == "xgb":
        if classes == 2:
            classifier = XGBClassifier(objective="binary:logistic")
        else:
            classifier = XGBClassifier(num_class=classes, objective="multi:softmax")

    # Resampling
    if resampler is None:
        X_train_resampled, y_train_resampled = X_train.copy(), y_train.copy()
    else:
        try:
            if resampler.get_params().get("n_neighbors") is not None:
                # Bound n_neighbours <= n_samples in each fold
                default = min(y_train.value_counts().min() - 1, resampler.get_params().get("n_neighbors"))  # type: ignore
                resampler.set_params(n_neighbors=default)
            if resampler.get_params().get("k_neighbors") is not None:
                default = min(y_train.value_counts().min() - 1, resampler.get_params().get("k_neighbors"))  # type: ignore
                resampler.set_params(k_neighbors=default)
            if resampler.get_params().get("enn") is not None:
                default = min(y_train.value_counts().min() - 1, resampler.get_params().get("enn").get_params().get("n_neighbors"))  # type: ignore
                resampler.set_params(enn=EditedNearestNeighbours(sampling_strategy="majority", n_neighbors=default))
            if resampler.get_params().get("smote") is not None:
                default = min(
                    y_train.value_counts().min() - 1, resampler.get_params().get("smote").get_params().get("k_neighbors")
                )  # type
                resampler.set_params(smote=SMOTE(k_neighbors=default))
            X_train_resampled, y_train_resampled = resampler.fit_resample(X_train, y_train)  # type: ignore
            resampled = 0
        except (ValueError, RuntimeError) as e:
            if DEBUG:
                print(f"Error resampling {classifier_name} with {resampler_name}")
                print(e)
                print(file_name)
                print(f"Internal Fold Count: {internal_fold_count}")
                print(pd.Series(y_train).value_counts())
                raise e
            else:
                return None

    # Train the classifier and predict the target variable
    selected_search_space = HYPERPARAMETERS[classifier_name]
    try:
        if "n_neighbors" in selected_search_space:
            min_samples = min(y_train.value_counts().min() - 1, 10)  # Bound n_neighbours <= n_samples in each fold
            selected_search_space["n_neighbors"] = Integer(1, min_samples)

        bayes_search = BayesSearchCV(
            estimator=classifier,
            search_spaces=selected_search_space,
            cv=StratifiedKFold(n_splits=internal_fold_count),
            refit=True,
            n_jobs=10,
            n_iter=10,
        )
        # bayes_search.fit(X_train_resampled, y_train_resampled)
        bayes_search = classifier.fit(X_train_resampled, y_train_resampled)  # type: ignore
        y_pred = bayes_search.predict(X_test)
    except Exception as e:
        if DEBUG:
            print(f"Error training {classifier_name} with {resampler_name}")
            print("Selected search space:")
            print(selected_search_space)
            print(e)
            print(file_name)
            print(internal_fold_count)
            print(pd.Series(y_train_resampled).value_counts())  # type: ignore
            print("\n\n\n")
            raise e
        else:
            return None

    # Don't break if we're using a classifier that doesn't have predict_proba
    if hasattr(bayes_search, "predict_proba"):
        lr_probs = bayes_search.predict_proba(X_test)
    else:
        # Assign the predicted probabilities to the y_pred variable
        lr_probs = np.zeros((len(y_pred), 2))
        for i, pred in enumerate(y_pred):
            lr_probs[i][pred] = 1

    recall: float = -1.0
    precision: float = -1.0
    f1: float = -1.0
    geometric_mean: float = -1.0
    roc_auc: float = -1.0
    pr_auc: float = -1.0
    cwa: float = -1.0

    # Multiclass PR AUC is not supported
    if classes > 2:
        pr_auc = -1
        roc_auc = -1
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")
        geometric_mean = geometric_mean_score(y_test, y_pred, average="multiclass") # type: ignore
        f1 = f1_score(y_test, y_pred, average="macro")
    else:
        precision = precision_score(y_test, y_pred, average="binary")
        recall = recall_score(y_test, y_pred, average="binary")
        geometric_mean = geometric_mean_score(y_test, y_pred, average="binary") # type: ignore
        f1 = f1_score(y_test, y_pred, average="binary")
        try:
            precision_, recall_, _ = precision_recall_curve(y_true=y_test.to_numpy(), probas_pred=lr_probs[:, 1])
            pr_auc = auc(recall_, precision_)
        except Exception:
            pr_auc = 0.5
        try:
            roc_auc = float(roc_auc_score(y_test.to_numpy(), lr_probs[:, 1]))
        except Exception:
            roc_auc = 0.5

    class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_test), y=y_test)
    recall_per_class, _, _, _ = precision_recall_fscore_support(y_test, y_pred, average=None, labels=np.unique(y_test))
    cwa = float(np.average(recall_per_class, weights=class_weights))
    ideal_hyperparameters = bayes_search.best_params_ if hasattr(bayes_search, "best_params_") else {}
    print(f"Accuracy: {accuracy_score(y_test, y_pred)} for {classifier_name} with {resampler_name} on {file_name}")

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "geometric_mean": geometric_mean,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "cwa": cwa,
        "learner": classifier_name,
        "resampler": resampler_name,
        "ideal_hyperparameters": ideal_hyperparameters,
    }


def train(args: tuple) -> pd.DataFrame:
    """
    Train the classifiers and calculate the metafeatures for the dataset.
    This function is used to parallelize the training process.
    It generates the training data for a single dataset across the search space.
    Do not call this except for when you want to create the training dataset.
    Arguments are passed in as a tuple.

    Args:
        file (Path): The path to the dataset file.
        classifier_name (str): The name of the classifier to use.
        resampler_name (str): The name of the resampler to use.

    Returns:
        pd.DataFrame: A DataFrame containing the calculated metafeatures and training results.
    """
    file = args[0]
    classifier_name = args[1]
    resampler_name = args[2]
    file_name = file.stem

    metafeatures = calculate_metafeatures(file)
    dataset = pd.read_csv(file)

    # Target variable is defined as the last column in the dataset
    y: pd.Series = pd.Series(dataset.iloc[:, -1].copy())
    X: pd.DataFrame = pd.DataFrame(dataset.iloc[:, :-1].copy().astype("int"))

    cross_validation = StratifiedKFold(
        n_splits=2,
        shuffle=True,
        random_state=42,
    )

    training_results = pd.DataFrame()

    for _, (train_index, test_index) in enumerate(cross_validation.split(X, y)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        scores: Dict[str, float | str] | None = train_classifiers(
            X_train.copy(),
            y_train.copy(),
            X_test.copy(),
            y_test.copy(),
            classifier_name,
            resampler_name,
            file_name,
        )
        if scores is not None:
            scores_df: pd.DataFrame = pd.DataFrame.from_dict([scores])  # type: ignore
            training_results = pd.concat(
                [training_results, scores_df],
                ignore_index=True,
            )

    return pd.concat([pd.DataFrame([metafeatures]), training_results], axis=1, ignore_index=False, sort=False, join="inner")


def get_existing_solutions(file: Path) -> list[tuple]:
    try:
        df = pd.read_csv(file)
        return list(df[['dataset', 'learner', 'resampler']].apply(lambda x: (x[0], x[1], x[2]), axis=1))
    except Exception:
        return []


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--debug", action="store_true")
    argparser.add_argument('--quiet', action='store_true')
    DEBUG = argparser.parse_args().debug
    QUIET = argparser.parse_args().quiet

    metafeature_file = Path("metafeatures.csv")
    os.system(f"touch {metafeature_file.absolute()}")
    os.system("rm ../data/processed_datasets/*")

    datasets = get_datasets()
    existing_solutions = get_existing_solutions(metafeature_file)
    combinations = itertools.product(datasets, get_classifiers().keys(), get_resamplers().keys())
    combinations = [c for c in combinations if (c[0].stem, c[1], c[2]) not in existing_solutions]

    # Calculate metafeatures and classifiers for each dataset
    mutex = Lock()
    with ThreadPoolExecutor(max_workers=1) as p:
        for i, result in enumerate(p.map(train, combinations)):
            try:
                metafeatures = pd.read_csv(metafeature_file)
            except Exception as e:
                metafeatures = pd.DataFrame()

            try:
                metafeatures = pd.concat([metafeatures, result], ignore_index=True)
            except pd.errors.EmptyDataError:
                metafeatures = result

            metafeatures.to_csv(metafeature_file, index=False)

    print("Done!")
