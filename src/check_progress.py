import pandas as pd
from create_metafeatures import get_datasets, get_classifiers, get_resamplers, get_existing_solutions
import itertools
from pathlib import Path


if __name__ == "__main__":
    datasets = get_datasets()
    existing_solutions = get_existing_solutions(Path("merged_metafeatures.csv"))
    combinations = itertools.product(datasets, get_classifiers().keys(), get_resamplers().keys())
    combinations = [c for c in combinations if (c[0].stem, c[1], c[2]) not in existing_solutions]
    combinations = [(c[0].stem, c[1], c[2]) for c in combinations]

    print(f"Total number of combinations: {len(combinations)}")

    missing_datasets = [d for d in datasets if d not in combinations[:][0]]
    missing_datasets = set(missing_datasets)
    print(f"Number of missing datasets: {len(missing_datasets)}")

    df = pd.DataFrame(combinations, columns=["dataset", "classifier", "resampler"])
    print(df)

    df['classifier'].value_counts().plot(kind='bar', title='Classifiers')




