from complexity import (
    complexity,
    weighted_complexity,
    dualweighted_complexity,
    tomelink_complexity,
)
from typing import Dict
from pathlib import Path
from distance import distance
from kmeans import KMeansMetadata
from hypersphere import create_hypersphere
from overlapping import volume_overlap
import numpy as np
import pandas as pd

def get_imbalance_ratio(y: pd.Series) -> float:
    return y.value_counts().min() / y.value_counts().max()

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

    kmeans_metadata: KMeansMetadata = KMeansMetadata().fit(X, y)  # Calculate unsupervised KMeans metadata
    hypershpere_metadata: Dict[str, float] = calculate_hypershpere_metadata(X, y)  # Calculate classification metadata
    complexity_metadata: Dict[str, float] = calculate_complexity_metadata(dataset)
    overlap: float = volume_overlap(X.values, y.values)
    imbalance_ratio: float = get_imbalance_ratio(y)

    metafeatures_dict: dict = {"imbalance_ratio": imbalance_ratio, "overlap": overlap, "dataset": file_name, "rows": X.shape[0]}
    metafeatures_dict.update(hypershpere_metadata)
    metafeatures_dict.update(kmeans_metadata.__dict__)
    metafeatures_dict.update(complexity_metadata)

    return metafeatures_dict



