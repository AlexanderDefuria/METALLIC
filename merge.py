import pandas as pd
from pathlib import Path
from glob import glob

dataframes = []
for file in glob("metafeatures*.csv"):
    dataframes.append(pd.read_csv(file))

pd.concat(dataframes).to_csv("merged_metafeatures.csv", index=False)


