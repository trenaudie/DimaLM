import os
import pandas as pd
import numpy as np
from pathlib import Path
from itertools import chain

ROOT_DIR = next(
    filter(
        lambda s: "LLM" in s.name, chain(Path().absolute().parents, [Path(os.getcwd())])
    ),
    None,
)
print(f"ROOT_DIR {ROOT_DIR}")
import configparser
import matplotlib.pyplot as plt

config = configparser.ConfigParser()
config.read(ROOT_DIR / "config.ini")
DATA_FOLDER = Path(config.get("paths", "path_data"))
print(f"DATA_FOLDER {DATA_FOLDER}")
assert os.path.isdir(DATA_FOLDER)
assert os.path.isdir(ROOT_DIR)
import sys

sys.path.append(str(ROOT_DIR))
from pathlib import Path
from time import time
from utils.io_utils import write_headlines_to_file
from tqdm import tqdm


def apply_function(df: pd.DataFrame):
    # apply this on the df after the groupby
    lp = 0
    rp = 0
    concats = pd.Series(["_"] * df.shape[0], index=df.index)
    while rp < df.shape[0]:
        if rp - lp <= 3:
            concats.iloc[rp] = "\n-".join(df.iloc[lp:rp]["headline"].astype(str).values)
            rp += 1
        else:
            lp += 1
    return concats


if __name__ == "__main__":
    path_df_news_context_done = ROOT_DIR / "temp/data_v5_company_context_done.parquet"
    if not path_df_news_context_done.exists():
        raise Exception("run data_v5_company_context.py first")
    start = time()
    dfnews = pd.read_parquet(path_df_news_context_done)
    dfnews = dfnews.loc[
        dfnews.index.get_level_values(0).isin(
            dfnews.index.get_level_values(0).unique()[:30]
        )
    ]  # REMOVE THIS CELL WHEN DONE DEBUGGING
    dfnews = dfnews.reset_index().set_index(["SYMBOL", "DATE"]).sort_index()
    dfnews["news_context"] = "None"
    tqdm.pandas()
    dfnews["news_context"] = (
        dfnews.groupby("SYMBOL")
        .progress_apply(apply_function)
        .reset_index(level=0, drop=True)
    )
    print(f"headlines concat finished in {time()-start:.2f} seconds")
    dfnews.to_parquet(ROOT_DIR / "temp/data_v5_context_news_done.parquet")
    write_headlines_to_file(
        dfnews=dfnews,
        x_col="context",
        add_cols=["company_name", "news_context"],
        filepath=ROOT_DIR / "data_preprocess" / "news_headlines_v5_concat.txt",
        limit=1000000,
        step=100,
        keep_date=False,
    )
    print(f"done")
