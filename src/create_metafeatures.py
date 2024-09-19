from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
from typing import Dict, List
import copy

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
from data_preprocessing import preprocess
from scipy.io import arff
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

from distance import distance
from kmeans import KMeansMetadata
from hypersphere import create_hypersphere
from overlapping import volume_overlap
from complexity import (
    complexity,
    weighted_complexity,
    dualweighted_complexity,
    tomelink_complexity,
)
from hyperparameters import HYPERPARAMETERS


def dataset_dir() -> Path:
    return Path(__file__).parent.parent / "data"


def arrf_datasets_dir() -> Path:
    return Path(__file__).parent.parent / "data" / "arrf_datasets"


def processed_datasets_dir() -> Path:
    path = Path(__file__).parent.parent / "data" / "processed_datasets"
    if not path.exists():
        os.makedirs(path)
    return path


def raw_datasets_dir() -> Path:
    return Path(__file__).parent.parent / "data" / "raw_datasets"


def collect_datasets(directory: Path) -> List[Path]:
    if not directory.exists() or not directory.is_dir():
        return []

    return list(directory.glob("*.csv"))


def collect_arrf_datasets() -> List[Path]:
    """
    Collects ARFF datasets and converts them to CSV format if they have not been converted before.
    Returns:
        List[Path]: A list of paths to the converted CSV datasets.
    """

    existing_csv_datasets = collect_datasets(raw_datasets_dir())
    existing_csv_datasets = [dataset.stem for dataset in existing_csv_datasets]
    arrf_datasets = list(arrf_datasets_dir().glob("*.arff"))
    arrf_to_process = [dataset for dataset in arrf_datasets if dataset.stem not in existing_csv_datasets]
    converted_paths = []

    for dataset in arrf_to_process:
        data, _ = arff.loadarff(dataset)
        df = pd.DataFrame(data)
        path = (raw_datasets_dir() / dataset.stem).with_suffix(".csv")
        print(f"Converting {dataset} to {path}")
        df.to_csv(path, index=False)
        converted_paths.append(path)

    return converted_paths


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
        # "xgb": XGBClassifier(),
        "knn": KNeighborsClassifier(),
        "dt": DecisionTreeClassifier(),
        "gnb": GaussianNB(),
        "svm": SVC(probability=True),
        "rf": RandomForestClassifier(),
        "ada": AdaBoostClassifier(),
        # "cat": CatBoostClassifier(verbose=0),
    }


def get_resamplers() -> Dict[str, object]:
    """
    Returns a dictionary of resamplers.
    Returns:
        Dict[str, object]: A dictionary where the keys are the names of the resamplers and the values are the corresponding resampler objects.
    """

    return {
        "none": None,
        "smote": SMOTE(random_state=42),
        # "smoteenn": SMOTEENN(random_state=42),
        "randomoversampler": RandomOverSampler(random_state=42),
        "adasyn": ADASYN(random_state=42),
        "borderlinesmote": BorderlineSMOTE(random_state=42),
        "smotesvm": SVMSMOTE(random_state=42),
        "randomundersampler": RandomUnderSampler(random_state=42),
        "clustercentroids": ClusterCentroids(random_state=42),
        "nearmissv1": NearMiss(version=1),
        "nearmissv2": NearMiss(version=2),
        "nearmissv3": NearMiss(version=3),
        "tomeklinks": TomekLinks(),
        "editednn": EditedNearestNeighbours(),
        "repeatededitednn": RepeatedEditedNearestNeighbours(),
        "allknn": AllKNN(),
        "condensednn": CondensedNearestNeighbour(random_state=42),
        "ncl": NeighbourhoodCleaningRule(),
        "instancehardness": InstanceHardnessThreshold(random_state=42),
        "smotetomek": SMOTETomek(random_state=42),
    }


def train_classifiers(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    classifier,
    resampler,
    classifier_name: str,
    resampler_name: str,
) -> Dict[str, float | str] | None:
    """
    Resample the dataset and train the classifier

    Returns the classifier's f1 score, accuracy, geometric mean, precision, recall, roc auc, pr auc, balanced accuracy, and class weighted accuracy
    """
    np.int = np.int32  # type: ignore
    np.float = np.float64  # type: ignore
    np.bool = np.bool_  # type: ignore
    internal_fold_count = 5

    # Scale the dataset
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    selected_search_space = {}

    # Get the minimum number of samples in each class
    min_samples = min(
        y_train.value_counts().min(),
        y_test.value_counts().min(),
        5,
    )

    # Clone the resampler object
    if resampler is not None:
        resampler = copy.deepcopy(resampler)
        if "k_neighbors" in resampler.get_params():
            resampler.set_params(k_neighbors=min_samples)
        if "smote" in resampler.get_params():
            resampler.set_params(smote=SMOTE(k_neighbors=min_samples))

    # Train the classifier and predict the target variable
    try:
        # Resample the dataset
        if resampler is not None:
            try:
                X_train, y_train = resampler.fit_resample(X_train, y_train)
            except (ValueError, RuntimeError):
                return None

        # Select the search space for the classifier
        selected_search_space = HYPERPARAMETERS[classifier_name]
        if "n_neighbors" in selected_search_space:
            # max_neighbours: int = (X_train.shape[0] // internal_fold_count) - 1
            # selected_search_space["n_neighbors"] = Integer(1, max_neighbours)  # Bound n_neighbours <= n_samples in each fold
            if min_samples < 2: return None
            selected_search_space["n_neighbors"] = Integer(1, min_samples)  # Bound n_neighbours <= n_samples in each fold

        # bayes_search = BayesSearchCV(
        #    estimator=classifier,
        #    search_spaces=selected_search_space,
        #    cv=internal_fold_count,
        #    refit=True,
        #    n_jobs=1,
        #    # verbose=10000,
        # )
        bayes_search = classifier
        bayes_search.fit(X_train, y_train)
        y_pred = bayes_search.predict(X_test)
    except Exception as e:
        print(f"Error training {classifier_name} with {resampler_name}")
        print("Selected search space:")
        print(selected_search_space)
        print(e)
        print("\n\n\n")
        raise e
        return {
            "accuracy": -1,
            "balanced_accuracy": -1,
            "recall": -1,
            "precision": -1,
            "f1": -1,
            "geometric_mean": -1,
            "roc_auc": -1,
            "pr_auc": -1,
            "cwa": -1,
        }

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
    if (len(np.unique(y_train)) + len(np.unique(y_test))) > 2:
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
        except Exception:
            pr_auc = 0.5
        try:
            roc_auc = float(roc_auc_score(y_test.to_numpy(), lr_probs[:, 1]))
        except Exception:
            roc_auc = 0.5

    class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_test), y=y_test)
    recall_per_class, _, _, _ = precision_recall_fscore_support(y_test, y_pred, average=None, labels=np.unique(y_test))
    cwa = float(np.average(recall_per_class, weights=class_weights))

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
        "classifier": classifier_name,
        "resampler": resampler_name,
    }


def calculate_metafeatures(file: Path) -> pd.DataFrame:
    """
    Calculate metafeatures for a given dataset file.
    Args:
        file (Path): The path to the dataset file.
    Returns:
        pd.DataFrame: A DataFrame containing the calculated metafeatures.
    Raises:
        Exception: If an error occurs during the calculation.
    """
    file_name = file.stem
    dataset = pd.read_csv(file)

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
                scores: Dict[str, float | str] | None = train_classifiers(
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    classifier,
                    resampler,
                    classifier_name,
                    resampler_name,
                )
                if scores is not None:
                    scores_df: pd.DataFrame = pd.DataFrame.from_dict([scores])  # type: ignore
                    training_metadata = pd.concat(
                        [training_metadata, scores_df],
                        ignore_index=True,
                    )

    # Calculate average scores
    results = training_metadata.groupby(["classifier", "resampler"]).mean().reset_index()
    results = results.assign(**hypershpere_metadata)
    results = results.assign(**kmeans_metadata.__dict__)
    results = results.assign(**complexity_metadata)
    results["imbalance_ratio"] = imbalance_ratio
    results["rows"] = X.shape[0]
    results["overlap"] = overlap
    results["dataset"] = file_name
    results = results.reindex(columns=reversed(results.columns))

    return results


if __name__ == "__main__":
    metafeature_file = Path("metafeatures.csv")
    existing_datasets = []
    if metafeature_file.exists():
        # os.system(f"rm {metafeature_file.absolute()}")
        try:
            existing_datasets = pd.read_csv(metafeature_file)["dataset"]
        except Exception:
            existing_datasets = []
    # os.system(f"touch {metafeature_file.absolute()}")

    # Convert ARFF datasets to CSV format and place in the raw_datasets directory
    collect_arrf_datasets()

    # Clean the raw CSV datasets and preprocess them
    datasets = collect_datasets(raw_datasets_dir())
    for dataset in tqdm(datasets, desc="Preprocessing Datasets to ../data/processed_datasets"):
        preprocess(dataset).to_csv(processed_datasets_dir() / dataset.name, index=False)

    # Existing Metafeature Datasets
    # metafeatures = pd.read_csv(metafeature_file)['dataset']
    # datasets = collect_datasets(processed_datasets_dir())
    # to_skip = [dataset.stem for dataset in datasets if dataset.stem in metafeatures]
    to_skip = []

    # Calculate metafeatures and classifiers for each dataset
    with ThreadPoolExecutor(max_workers=1) as executor:
        datasets = collect_datasets(processed_datasets_dir())
        datasets = datasets[:40]
        datasets.sort(key=lambda p: p.name.lower())
        datasets = [dataset for dataset in datasets if dataset.stem not in to_skip]
        datasets = [dataset for dataset in datasets if dataset.stem not in existing_datasets]

        with tqdm(
            executor.map(calculate_metafeatures, datasets),
            total=len(datasets),
            desc="Calculating Metafeatures",
        ) as tbar:
            for result in list(tbar):
                try:
                    metafeatures = pd.read_csv(metafeature_file)
                except Exception:
                    metafeatures = pd.DataFrame()

                try:
                    tbar.set_description(f"Calculating Metafeatures for {result['dataset']}")
                    metafeatures = pd.concat([metafeatures, result], ignore_index=True)
                    print("Trained")
                except pd.errors.EmptyDataError:
                    print("Empty Data Error")
                    metafeatures = result
                    # print(metafeatures)

                metafeatures.to_csv(metafeature_file, index=False)

    print("Done!")
