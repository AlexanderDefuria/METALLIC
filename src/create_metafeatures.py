import multiprocessing as mp
from pathos.pools import ProcessPool
import os
import fcntl
import time
from contextlib import contextmanager
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
import os
import sys

def print(*args, **kwargs):
    # Convert all arguments to strings and join them
    message = ' '.join(str(arg) for arg in args)
    
    # Use os.system to echo the message
    # os.system(f'echo {message}')
    
    # Print to stdout using the original sys.stdout.write
    sys.stdout.write(message + '\n')
    sys.stdout.flush()


def get_classifiers(choice: str | None = None) -> Dict[str, object]:
    classifiers = {
        "xgb": XGBClassifier(),
        "knn": KNeighborsClassifier(),
        "dt": DecisionTreeClassifier(),
        "gnb": GaussianNB(),
        "svm": SVC(probability=True),
        "rf": RandomForestClassifier(),
        "ada": AdaBoostClassifier(),
        "cat": CatBoostClassifier(verbose=0, task_type='CPU', thread_count=1),
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
    quick: bool = False,
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
            if False: #DEBUG:
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
    
    starting_class_label = np.concatenate((y_train, y_test, y_train_resampled)).min()
    print(f"STARTING CLASS {starting_class_label}")

    print(f"Training {file_name} - {classifier_name} - {resampler_name}")
    try:
        if "n_neighbors" in selected_search_space:
            min_samples = min(y_train.value_counts().min() - 1, 10)  # Bound n_neighbours <= n_samples in each fold
            selected_search_space["n_neighbors"] = Integer(1, min_samples)

        model = BayesSearchCV(
            estimator=classifier,
            search_spaces=selected_search_space,
            cv=StratifiedKFold(n_splits=internal_fold_count),
            refit=True,
            n_jobs=1,
        ) # if not quick else classifier
        model.fit(X_train_resampled, y_train_resampled)  # type: ignore
        y_pred = model.predict(X_test)
    except Exception as e:
        print(f"Error training {classifier_name} with {resampler_name}")
        if False: # DEBUG:
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
    if hasattr(model, "predict_proba"):
        lr_probs = model.predict_proba(X_test)
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
    ideal_hyperparameters = model.best_params_ if hasattr(model, "best_params_") else {}
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
    file_name = args[0]
    classifier_name = args[1]
    resampler_name = args[2]
    quick = args[3]
    dataset = args[4]

    print(f"Started {file_name} - {classifier_name} - {resampler_name}")
    
    # Target variable is defined as the last column in the dataset
    metafeatures = calculate_metafeatures(file_name, dataset)
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
            quick,
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
    except Exception as e:
        print("No solutions")
    return [] 


if __name__ == "__main__":
    os.environ['OMP_NUM_THREADS'] = '1'
    start_time = time.time()
    metallic_dir = Path(__file__).parent.parent.resolve()
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--debug", action="store_true")
    argparser.add_argument("--quick", action="store_true")
    argparser.add_argument("--cpu", default=1)
    argparser.add_argument("--slurmid", default=1)
    argparser.add_argument("--slurmcount", default=1)
    argparser.add_argument("--tempdir", default=metallic_dir)
    argparser.add_argument("--homedir", default=metallic_dir)

    DEBUG = argparser.parse_args().debug
    CPUS = int(argparser.parse_args().cpu)
    QUICK = argparser.parse_args().quick
    SLURMID = int(argparser.parse_args().slurmid)
    SLURMCOUNT = int(argparser.parse_args().slurmcount)
    home_dir = Path(argparser.parse_args().homedir)
    temp_dir = Path(argparser.parse_args().tempdir)
    data_dir = temp_dir / "data"
    out_dir = temp_dir
    assert CPUS >= 1
    print(f"SLURMID: {SLURMID} with {CPUS} CPUS")

    if SLURMCOUNT>1:
        metafeature_file = out_dir / f"metafeatures_{SLURMID}.csv"
        existing_solutions = get_existing_solutions(home_dir / "merged_metafeatures.csv")
    else:
        metafeature_file = out_dir / "metafeatures.csv"
        existing_solutions = get_existing_solutions(metafeature_file)

    datasets = sorted(get_datasets(data_dir))
    #datasets = [dataset for dataset in datasets if "geobia" in dataset.stem.lower()]
    combinations = itertools.product(datasets, get_classifiers().keys(), get_resamplers().keys(), [QUICK])
    combinations = [c for c in combinations if (c[0].stem, c[1], c[2]) not in existing_solutions]
    combinations = combinations[SLURMID-1::SLURMCOUNT]
    # combinations = combinations[:2]
    for i, c in enumerate(combinations):
        try:
            combinations[i] =  (c[0], c[1], c[2], c[3], pd.read_csv(c[0]))
        except Exception:
            combinations[i] = None
    combinations = [c for c in combinations if c is not None]
            
            
    ncpus = int(os.environ.get('SLURM_CPUS_PER_TASK',default=1)) * 4
    print(f"START COMPUTE for {len(combinations)} COMBINATIONS and {ncpus} CPUS")

    
    with mp.get_context("spawn").Pool(processes=ncpus) as pool:
        for result in pool.imap_unordered(train, combinations):
            if result is None:
                print("Failed Result...")
                continue
                                                                  
            if not metafeature_file.exists():
                result.to_csv(metafeature_file, index=False)
            else:
                f = open(metafeature_file, 'a')
                f.write(result.to_csv(index=False, header=False))
                f.close()

    print("Done!")
