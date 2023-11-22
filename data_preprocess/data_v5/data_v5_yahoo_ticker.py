# get the ticker name for every company name
from bs4 import BeautifulSoup
import requests
import re
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
from tqdm import tqdm
from collections import Counter
from utils.io_utils import write_headlines_to_file
from concurrent.futures import ThreadPoolExecutor


def get_ticker_from_name(company_name):
    # Use the Yahoo Finance search functionality
    search_url = f"https://finance.yahoo.com/lookup?s={company_name}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    response = requests.get(search_url, headers=headers)
    response.raise_for_status()

    soup = BeautifulSoup(response.content, "html.parser")

    # Try finding the ticker from search results
    ticker_anchor = soup.find_all(
        "a", href=re.compile("\/quote\/.*"), attrs={"data-symbol": True}
    )
    if ticker_anchor:
        ticker_anchor = ticker_anchor[0]
        data_symbol = ticker_anchor["data-symbol"]
        return data_symbol
    else:
        return None


if __name__ == "__main__":
    path_dfnews = ROOT_DIR / "temp/data_v5_context_news_done.parquet"
    assert path_dfnews.exists()
    dfnews = pd.read_parquet(path_dfnews)
    print(f"working with shape {dfnews.shape}")
    tqdm.pandas()
    company_names = dfnews["company_name"].unique()
    company_name_ticker = {}
    with ThreadPoolExecutor(max_workers=10) as executor:
        for company_name in tqdm(company_names):
            ticker = executor.submit(get_ticker_from_name, company_name)
            company_name_ticker[company_name] = ticker.result()
    dfnews["ticker"] = dfnews["company_name"].apply(lambda x: company_name_ticker[x])
    dfnews.to_parquet(ROOT_DIR / "temp/data_v5_ticker_done.parquet")
