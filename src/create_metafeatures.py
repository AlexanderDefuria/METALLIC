from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
from typing import Dict, List, Any
import copy
import itertools

from catboost import CatBoostClassifier
from imblearn.base import BaseSampler
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
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
    unique_classes: np.ndarray = y.unique()  # type: ignore

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


def get_classifiers(choice: str | None = None) -> Dict[str, object]:
    classifiers = {
        "xgb": XGBClassifier(),
        "knn": KNeighborsClassifier(),
        # "dt": DecisionTreeClassifier(),
        #"gnb": GaussianNB(),
        #"svm": SVC(probability=True),
        #"rf": RandomForestClassifier(),
        #"ada": AdaBoostClassifier(),
        #"cat": CatBoostClassifier(verbose=0),
    }
    return classifiers if choice is None else classifiers[choice]


def get_resamplers(choice: str | None = None) -> Dict[str, object]:
    """
    Returns a dictionary of resamplers.
    Returns:
        Dict[str, object]: A dictionary where the keys are the names of the resamplers and the values are the corresponding resampler objects.
    """
    resamplers = {
        "none": None,
        # "smote": SMOTE(random_state=42),
        # "smoteenn": SMOTEENN(random_state=42),
        # "randomoversampler": RandomOverSampler(random_state=42),
        # "adasyn": ADASYN(random_state=42, n_neighbors=5),
        #"borderlinesmote": BorderlineSMOTE(random_state=42),
        #"smotesvm": SVMSMOTE(random_state=42),
        "randomundersampler": RandomUnderSampler(random_state=42, sampling_strategy="majority"),
        #"clustercentroids": ClusterCentroids(random_state=42),
        #"nearmissv1": NearMiss(version=1),
        #"nearmissv2": NearMiss(version=2),
        #"nearmissv3": NearMiss(version=3),
        #"tomeklinks": TomekLinks(),
        #"editednn": EditedNearestNeighbours(sampling_strategy="majority", kind_sel="mode"),
        #"repeatededitednn": RepeatedEditedNearestNeighbours(sampling_strategy="majority", kind_sel="mode"),
        #"allknn": AllKNN(sampling_strategy="majority", kind_sel="mode"),
        #"condensednn": CondensedNearestNeighbour(random_state=42, sampling_strategy="majority"),
        #"ncl": NeighbourhoodCleaningRule(sampling_strategy="majority", threshold_cleaning=0.2),
        #"instancehardness": InstanceHardnessThreshold(random_state=42),
        #"smotetomek": SMOTETomek(random_state=42),
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

    # Get the classifier and resampler
    classifier: BaseEstimator = get_classifiers()[classifier_name] # type: ignore
    resampler: BaseSampler | None = get_resamplers()[resampler_name] # type: ignore


    # Scale the dataset
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    classes = np.concatenate((y_train, y_test)).max() + 1

    # class_counts = pd.Series(np.concatenate((y_train, y_test))).value_counts()
    class_counts = pd.Series(y_train, y_test).value_counts()
    min_class_count = class_counts.min()  # How many instances of the least frequent class are there?
    internal_fold_count = min(2, min_class_count)
    selected_search_space = {}

    # Get the minimum number of samples in each class
    min_samples = min(y_train.value_counts().min() - 1, 10)

    # Clone the resampler object
    if resampler is not None:
        resampler  = copy.deepcopy(resampler)
        if "k_neighbors" in resampler.get_params():
            resampler.set_params(k_neighbors=min_samples)
        if "smote" in resampler.get_params():
            resampler.set_params(smote=SMOTE(k_neighbors=min_samples))
        if "n_neighbors" in resampler.get_params():
            resampler.set_params(n_neighbors=min_samples)

    if classifier_name == "xgb":
        if classes == 2:
            classifier = XGBClassifier(objective="binary:logistic")
        else:
            classifier = XGBClassifier(num_class=classes, objective="multi:softmax")

    # Train the classifier and predict the target variable
    try:
        # Resample the dataset
        if resampler is not None:
            try:
                X_train_resampled, y_train_resampled = resampler.fit_resample(X_train, y_train) # type: ignore
            except (ValueError, RuntimeError) as e:
                # Sometimes a resampelr is not well suited to a dataset.
                # The initial cause of this exception was ADASYN not having appropriate
                # nearest neighbours to resample.
                print("resampling error")
                print(e)
                return None
        else:
            X_train_resampled, y_train_resampled = X_train.copy(), y_train.copy()

        selected_search_space = HYPERPARAMETERS[classifier_name]
        if "n_neighbors" in selected_search_space:
            # Bound n_neighbours <= n_samples in each fold
            selected_search_space["n_neighbors"] = Integer(1, min_samples )

        # print(f"Training {classifier_name} with {resampler_name} for {file_name} using {internal_fold_count} CFV")
        bayes_search = BayesSearchCV(
            estimator=classifier,
            search_spaces=selected_search_space,
            cv=StratifiedKFold(n_splits=internal_fold_count, shuffle=True, random_state=42),
            refit=True,
            n_jobs=1,
        )
        # bayes_search.fit(X_train_resampled, y_train_resampled)
        bayes_search = classifier.fit(X_train_resampled, y_train_resampled)
        y_pred = bayes_search.predict(X_test)
    except Exception as e:
        print(f"Error training {classifier_name} with {resampler_name}")
        print("Selected search space:")
        print(selected_search_space)
        print(e)
        print(file_name)
        print(internal_fold_count)
        print(pd.Series(y_train_resampled).value_counts()) # type: ignore
        print("\n\n\n")
        return None
        # raise e

    # Don't break if we're using a classifier that doesn't have predict_proba
    if hasattr(bayes_search, "predict_proba"): lr_probs = bayes_search.predict_proba(X_test)
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
    ideal_hyperparameters = bayes_search.best_params_ if hasattr(bayes_search, "best_params_") else {}
        
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
        "ideal_hyperparameters": ideal_hyperparameters,
    }


# def calculate_metafeatures(file: Path, classifier_name: str, resampler_name: str) -> pd.DataFrame:
def calculate_metafeatures(file: Path) -> dict:
    """
    Calculate metafeatures for a given dataset file. 
    This is used as input for the recommender system models.
    Call this function during the training data creation process 
    as well as from the recommender system during inference.
    Args:
        file (Path): The path to the dataset file.
    Returns:
        dict: A dictionary containing the calculated metafeatures.
    Raises:
        Exception: If an error occurs during the calculation.
    """

    file_name = file.stem
    dataset = pd.read_csv(file)

    # Target variable is defined as the last column in the dataset
    y: pd.Series = pd.Series(dataset.iloc[:, -1].copy())
    X: pd.DataFrame = pd.DataFrame(dataset.iloc[:, :-1].copy().astype("int"))

    kmeans_metadata: KMeansMetadata = KMeansMetadata().fit(X, y) # Calculate unsupervised KMeans metadata
    hypershpere_metadata: Dict[str, float] = calculate_hypershpere_metadata(X, y) # Calculate classification metadata
    complexity_metadata: Dict[str, float] = calculate_complexity_metadata(dataset)
    overlap: float = volume_overlap(X.values, y.values)
    imbalance_ratio: float = get_imbalance_ratio(y)


    metafeatures_dict: dict = {
        "imbalance_ratio": imbalance_ratio,
        "overlap": overlap,
        "dataset": file_name,
        "rows": X.shape[0]
    }
    metafeatures_dict.update(hypershpere_metadata)
    metafeatures_dict.update(kmeans_metadata.__dict__)
    metafeatures_dict.update(complexity_metadata)

    return metafeatures_dict



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


def get_existing_solutions() -> list:
    return []



if __name__ == "__main__":
    metafeature_file = Path("metafeatures.csv")
    existing_datasets = []
    if metafeature_file.exists():
        os.system(f"rm {metafeature_file.absolute()}")
        try:
            existing_datasets = pd.read_csv(metafeature_file)["dataset"]
        except Exception:
            existing_datasets = []
    os.system(f"touch {metafeature_file.absolute()}")
    os.system("rm ../data/processed_datasets/*")

    collect_arrf_datasets() # Convert ARFF datasets to CSV format and place in the raw_datasets directory
    datasets = collect_datasets(raw_datasets_dir()) # Clean the raw CSV datasets and preprocess them
    datasets = [dataset for dataset in datasets if dataset.stem not in ["primary-tumor"]] # Filter datasets
    to_skip = []

    for dataset in tqdm(datasets, desc="Preprocessing Datasets to ../data/processed_datasets"):
        df_processed: pd.DataFrame | None = preprocess(dataset)
        if df_processed is not None:
            if df_processed['cls'].value_counts().min() > 5:
                df_processed.to_csv(processed_datasets_dir() / dataset.name, index=False)


    datasets = collect_datasets(processed_datasets_dir())
    # datasets = [dataset for dataset in datasets if dataset.stem == "autos"]
    # datasets.sort(key=lambda p: p.name.lower())
    shuffle = np.random.permutation(len(datasets))
    datasets = [datasets[i] for i in shuffle]
    datasets = [dataset for dataset in datasets if dataset.stem not in to_skip]
    datasets = [dataset for dataset in datasets if dataset.stem not in existing_datasets]
    combinations = itertools.product(datasets, get_classifiers().keys(), get_resamplers().keys())

    ignore = get_existing_solutions()

    # Calculate metafeatures and classifiers for each dataset
    with ThreadPoolExecutor(max_workers=20) as executor:
        with tqdm(
            executor.map(train, combinations),
            total=len(datasets) * len(get_classifiers()) * len(get_resamplers()),
            desc="Calculating Metafeatures",
        ) as tbar:
            for result in tbar:
                try:
                    metafeatures = pd.read_csv(metafeature_file)
                except Exception:
                    metafeatures = pd.DataFrame()

                try:
                    metafeatures = pd.concat([metafeatures, result], ignore_index=True)
                except pd.errors.EmptyDataError:
                    metafeatures = result

                metafeatures.to_csv(metafeature_file, index=False)

    print("Done!")
