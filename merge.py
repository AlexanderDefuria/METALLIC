import pandas as pd
from pathlib import Path
from glob import glob

dataframes = []
for file in glob("./out/metafeatures*.csv"):
    dataframes.append(pd.read_csv(file))

merged = Path("./merged_metafeatures.csv")
if merged.exists():
    dataframes.append(pd.read_csv(merged))

df = pd.concat(dataframes)
df.set_index(["dataset", "learner", "resampler"], inplace=True)
#print(list(df.index))
df = df[~df.index.duplicated(keep='first')]
df = df.reset_index()
#print(list(df.columns))
df.to_csv("merged_metafeatures.csv", index=False)



print(f"Merged {len(df)} results")


