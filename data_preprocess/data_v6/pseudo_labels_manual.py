# %%
import os
import pandas as pd
import numpy as np
from pathlib import Path

ROOT_DIR = next(filter(lambda s: "LLM" in s.name, Path(os.getcwd()).parents))
print(ROOT_DIR)
import configparser
import matplotlib.pyplot as plt

config = configparser.ConfigParser()
config.read(ROOT_DIR / "config.ini")
DATA_FOLDER = Path(config.get("paths", "path_data"))
print(DATA_FOLDER)
assert os.path.isdir(DATA_FOLDER)
assert os.path.isdir(ROOT_DIR)
import sys

sys.path.append(str(ROOT_DIR))
pd.set_option("display.max_colwidth", None)


# %%
def print_some_rows_contexts():
    from itertools import islice

    for k in islice(range(60), 10, 20):
        print(f"headline {sadf['headline'].iloc[k]}")
        for i in range(5):
            print(
                sadf["news_cluster_last3"].iloc[k][i], sadf["ret10d_context"].iloc[k][i]
            )
        print(f"\n\n\n\n")


# %%
# how to pseudo label?
# manually label the first 5 of each cluster
def clustering_pseudolabel_manual(df: pd.DataFrame, start_index=0):
    # df is the result of a groupby clustering
    # df is sorted by date
    print(f"cluster {df['cluster'].iloc[0]}")
    assert "pseudo_label" in df.columns, "pseudo_label must exist"
    assert df.index.get_level_values("DATE").is_monotonic_increasing
    for k in range(6):
        idx = start_index + k
        newret = input(
            f"headline {df['headline'].iloc[idx]} - return {df['RET_5D_pos'].iloc[idx]} - {df['RET_10D_pos'].iloc[idx]}"
        )
        current_label = df["pseudo_label"].iloc[idx]
        if not np.isnan(current_label):
            print(f"already labelled {current_label}")
            continue
        if newret.lower().startswith("u"):
            # up prediction
            df["pseudo_label"].iloc[idx] = 1
        elif newret.lower().startswith("d"):
            # down prediction
            df["pseudo_label"].iloc[idx] = 0
        elif newret == "1":
            # up prediction
            df["pseudo_label"].iloc[idx] = 1
        elif newret == "0":
            # down prediction
            df["pseudo_label"].iloc[idx] = 0
        elif newret == "x":
            # exit
            break
        elif newret == "n" or newret == "2":
            # neutral
            df["pseudo_label"].iloc[idx] = 2
        else:
            print("invalid input")
            print(f"breaking at {idx}")
            raise ValueError
    return df


# %%
path_data = ROOT_DIR / "data_preprocess/data_v6/temp_df_clusters_v3.parquet"
# path_data = ROOT_DIR / "data_preprocess/data_v6/temp_pseudo_labels_manual_v1.0.parquet"
# this second dataframe has the context of past headlines, so we wont use it. We are now dynamically creating context.

assert path_data.exists()
df0 = pd.read_parquet(path_data)
df0.shape

# %%
df0.columns

for col in df0.columns:
    if "_replaced" in col or "_diff" in col or "embeddings" in col:
        df0 = df0.drop(col, axis=1)

# %%
# loading street account data
sadf = pd.read_parquet(f"temp_pseudo_labels_v1.0.parquet")
if "pseudo_label" not in sadf.columns:
    sadf["pseudo_label"] = np.nan


# %%
if "cluster" in sadf.index.names:
    if "cluster" in sadf.columns:
        sadf = sadf.reset_index(level="cluster", drop=True)
    else:
        raise NotImplementedError

print(f"shape of sadf: {sadf.shape}")
sadf = sadf.sort_values(by=["cluster", "DATE", "SYMBOL"])

sadf[["headline", "pseudo_label", "RET_5D_pos", "RET_10D_pos"]].head(10)
# %% sadf
sadf.loc[sadf.cluster == 3].head(10)[["headline", "RET_5D_pos", "RET_10D_pos"]]
# %%

sadf.loc[sadf.cluster == 0][["headline", "pseudo_label"]]

# %%
if "pseudo_label" not in sadf.columns:
    sadf["pseudo_label"] = np.nan
from tqdm import tqdm

# %%

sadf.columns
# %%
sadf.loc[sadf.cluster == 1][["headline", "pseudo_label"]].iloc[5:10]

# %%
tqdm.pandas()
sadf = sadf.groupby("cluster").progress_apply(
    lambda x: clustering_pseudolabel_manual(x, start_index=6)
)

# %%
# 6 and 7 to redo
sadf.loc[sadf.cluster == 0][["headline", "pseudo_label"]]

# %%
manual_labels = sadf.loc[sadf.pseudo_label.notna()]
print(f"number of manual labels {manual_labels.shape}")
manual_labels.iloc[::2, -4:]


# %%
sadf.to_parquet(f"temp_pseudo_labels_v1.0.parquet")
