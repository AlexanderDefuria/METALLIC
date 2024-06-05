from threading import RLock
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
from typing import Dict, List

from catboost import CatBoostClassifier
import pandas as pd
import numpy as np
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
from xgboost import XGBClassifier
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE, RandomOverSampler 
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

from distance import distance
from kmeans import KMeansMetadata
from hypersphere import create_hypersphere
from overlapping import volume_overlap
from complexity_metric import complexity
from weighted_complexity import weighted_complexity
from tomeklink_complexity import tomelink_complexity
from dual_weighted_complexity import dualweighted_complexity


def dataset_dir() -> Path:
    return Path(__file__).parent.parent / "datasets"


def collect_datasets() -> List[Path]:
    return list(dataset_dir().glob("*.csv"))


def calculate_complexity_metadata(data: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate the complexity metadata for the dataset

    Returns the following metadata:
    - complexity: The complexity of the dataset
    """

    # Calculate the complexity of the dataset
    complexity_score = complexity(data)
    weighted_complexity_score = weighted_complexity(data)
    dual_weighted_complexity_score = dualweighted_complexity(data)
    tomeklink_complexity_score = tomelink_complexity(data)
    number_of_classes = len(data.iloc[:, -1].unique())
    number_of_rows = data.shape[0]
    number_of_columns = data.shape[1]

    return {
        "complexity": complexity_score,
        "weighted_complexity": weighted_complexity_score,
        "dual_weighted_complexity": dual_weighted_complexity_score,
        "tomeklink_complexity": tomeklink_complexity_score,
        "number_of_classes": number_of_classes,
        "number_of_rows": number_of_rows,
        "number_of_columns": number_of_columns,
    }
        
        
def calculate_hypershpere_metadata(X: pd.DataFrame, y: pd.Series):
    y = y.astype("int")

    # Calculate number of unique classes and number of elements in each class
    unique_classes: np.ndarray = y.unique()
    class_counts: pd.Series = y.value_counts()

    # These are helper functions to calculate the metadata
    hyper_centres: np.ndarray = create_hypersphere(X.values, y.values)
    avg_distance_between_class: float = distance(hyper_centres, unique_classes)
    classes_hypershpere: list = list(set(hyper_centres[:, -1]))
    clusters_per_class = [len(hyper_centres[hyper_centres[:, -1] == c]) for c in classes_hypershpere]
    minority_class: int = np.min(clusters_per_class)
    majority_class: int = np.max(clusters_per_class)
    minority_class_index: int = clusters_per_class.index(minority_class)
    majority_class_index: int = clusters_per_class.index(majority_class)

    # These are saved as metadata in features.csv
    total_clusters: int = len(hyper_centres)
    average_samples_per_cluster: float = len(y) / total_clusters
    average_samples_per_cluster_majority: float = majority_class / total_clusters
    average_samples_per_cluster_minority: float = minority_class / total_clusters

    return {
        "samples_per_hypersphere": average_samples_per_cluster,
        "total_hypersheres": total_clusters,
        "average_distance_between_classes": avg_distance_between_class,
        "samples_per_hypersphere_minority": average_samples_per_cluster_minority,
        "samples_per_hypersphere_majority": average_samples_per_cluster_majority,
        "hypershpere_count_minority": clusters_per_class[minority_class_index],
        "hypershpere_count_majority": clusters_per_class[majority_class_index],
    }

def get_imbalance_ratio(y: pd.Series) -> float:
    return y.value_counts().min() / y.value_counts().max()


def get_classifiers() -> Dict[str, object]:
    return {
        'knn': KNeighborsClassifier(),
        "dt": DecisionTreeClassifier(),
        "gnb": GaussianNB(),
        "svm": SVC(probability=True),
        'rf': RandomForestClassifier(),
        'xgb': XGBClassifier(learning_rate=0.1, max_depth=3),
        'ada': AdaBoostClassifier(),
        'cat': CatBoostClassifier(verbose=0),
    }


def get_resamplers() -> Dict[str, object]:
    return {
        "none": None,
        "smote": SMOTE(),
        "smoteenn": SMOTEENN(random_state=42),
        "randomoversampler": RandomOverSampler(random_state=42),
        'adasyn': ADASYN(random_state=42),
        'borderlinesmote': BorderlineSMOTE(random_state=42),
        'smotesvm': SVMSMOTE(random_state=42),
        'randomundersampler': RandomUnderSampler(random_state=42),
        'clustercentroids': ClusterCentroids(random_state=42),
        'nearmissv1': NearMiss(version=1),
        'nearmissv2': NearMiss(version=2),
        'nearmissv3': NearMiss(version=3),
        'tomeklinks': TomekLinks(),
        'editednn': EditedNearestNeighbours(),
        'repeatededitednn': RepeatedEditedNearestNeighbours(),
        'allknn': AllKNN(),
        'condensednn': CondensedNearestNeighbour(random_state=42),
        'ncl': NeighbourhoodCleaningRule(),
        'instancehardness': InstanceHardnessThreshold(random_state=42),
        'smotetomek': SMOTETomek(random_state=42),
    }


def train_classifiers(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, classifier, resampler) -> Dict[str, float]:
    """
    Resample the dataset and train the classifier

    Returns the classifier's f1 score, accuracy, geometric mean, precision, recall, roc auc, pr auc, balanced accuracy, and class weighted accuracy
    """

    # Resample the dataset
    if resampler is not None:
        X_train, y_train = resampler.fit_resample(X_train, y_train)

    # Train the classifier and predict the target variable
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    # Don't break if we're using a classifier that doesn't have predict_proba
    if hasattr(classifier, "predict_proba"):
        lr_probs = classifier.predict_proba(X_test)
    else:
        lr_probs = np.zeros((len(y_train), 2))

    # Multiclass PR AUC is not supported
    if len(np.unique(y_train)) > 2:
        pr_auc = -1
        roc_auc = -1
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")
        geometric_mean = geometric_mean_score(y_test, y_pred, average="multiclass")
        f1 = f1_score(y_test, y_pred, average="macro")
    else:
        precision = precision_score(y_test, y_pred, average="binary")
        recall = recall_score(y_test, y_pred, average="binary")
        geometric_mean = geometric_mean_score(y_test, y_pred, average="binary")
        f1 = f1_score(y_test, y_pred, average="binary")
        try:
            precision_, recall_, _ = precision_recall_curve(y_true=y_test.to_numpy(), probas_pred=lr_probs[:, 1])
            pr_auc = auc(recall_, precision_)
        except Exception as e:
            pr_auc = 0.5
        try:
            roc_auc = roc_auc_score(y_test.to_numpy(), lr_probs[:, 1])
        except Exception as e:
            roc_auc = 0.5

    class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_test), y=y_test)
    recall_per_class, _, _, _ = precision_recall_fscore_support(y_test, y_pred, average=None, labels=np.unique(y_test))
    cwa = np.average(recall_per_class, weights=class_weights)
    

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "geometric_mean": geometric_mean,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "cwa": cwa,
    }


def calculate_metafeatures(file: Path, lock: RLock) -> pd.DataFrame:

    file_name = file.stem
    dataset = pd.read_csv(file)

    print(f"Processing {file_name}")

    # Target variable is defined as the last column in the dataset
    y: pd.Series = pd.Series(dataset.iloc[:, -1].copy())
    X: pd.DataFrame = pd.DataFrame(dataset.iloc[:, :-1].copy().astype("int"))

    # Calculate unsupervised KMeans metadata
    kmeans_metadata: KMeansMetadata = KMeansMetadata().fit(X, y)

    # Calculate classification metadata
    hypershpere_metadata: Dict[str, float] = calculate_hypershpere_metadata(X, y)
    complexity_metadata: Dict[str, float] = calculate_complexity_metadata(dataset)
    overlap: float = volume_overlap(X.values, y.values)
    imbalance_ratio: float = get_imbalance_ratio(y)

    # Score classifiers
    cross_validation = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    training_metadata = pd.DataFrame()

    for _, (train_index, test_index) in enumerate(cross_validation.split(X, y)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        for classifier_name, classifier in get_classifiers().items():
            for resampler_name, resampler in get_resamplers().items():
                scores = train_classifiers(X_train, y_train, X_test, y_test, classifier, resampler)
                scores["classifier"] = classifier_name
                scores["resampler"] = resampler_name
                try:
                    training_metadata = pd.concat([training_metadata, pd.DataFrame.from_dict([scores])], ignore_index=True)
                except Exception as e:
                    print(e)
                    print(scores)
                    print(training_metadata)
                    raise e
                
    # Calculate average scores
    results = training_metadata.groupby(["classifier", "resampler"]).mean().reset_index()
    results = results.assign(**hypershpere_metadata)
    results = results.assign(**kmeans_metadata.__dict__)
    results = results.assign(**complexity_metadata)
    results["imbalance_ratio"] = imbalance_ratio
    results["rows"] = X.shape[0]
    results["overlap"] = overlap
    results['dataset'] = file_name
    results = results.reindex(columns=reversed(results.columns))
    
    with lock: 
        try: 
            metafeatures = pd.read_csv("features_new.csv")
            metafeatures = pd.concat([metafeatures, results], ignore_index=True)
        except pd.errors.EmptyDataError:
            metafeatures = results
        except Exception as e:
            print(e)
            print(results)
            print(metafeatures)
            raise e
        
        metafeatures.to_csv("features_new.csv", index=False)
    
    return file_name


if __name__ == "__main__":
    os.remove("features_new.csv") 
    os.system("touch features_new.csv") 

    lock = RLock()
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        datasets = sorted(collect_datasets())
        executor.map(calculate_metafeatures, datasets, [lock] * len(datasets))
    
    print("Done!")
