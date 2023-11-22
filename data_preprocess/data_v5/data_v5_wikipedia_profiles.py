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
from json import dump

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
import spacy
from collections import Counter
import requests
from utils.io_utils import write_headlines_to_file
from concurrent.futures import ThreadPoolExecutor
import wikipediaapi
from time import time

wikipediaapi.logging.getLogger().handlers = []
import spacy

wiki_wiki = wikipediaapi.Wikipedia("DataV5", "en")


def get_summary_company_name(company_name):
    nlp = spacy.load("en_core_web_sm")
    page_py = wiki_wiki.page(company_name)
    page_summary = " ".join([sent.text for sent in nlp(page_py.summary).sents][:3])
    if len(page_summary) < 5:
        if "Company" not in company_name:
            company_name = f"{company_name} Company"
            print(f"retrying with {company_name}")
            page_summary = get_summary_company_name(company_name)
        else:
            return ""
    return page_summary


dfnews = pd.read_parquet(ROOT_DIR / "temp/data_v5_company_name_done.parquet")
start = time()
company_names = {}
with ThreadPoolExecutor(max_workers=10) as executor:
    for i, company_name in enumerate(tqdm(dfnews["company_name"].unique())):
        company_name_future = executor.submit(get_summary_company_name, company_name)
        try:
            company_names[company_name] = company_name_future.result()
            if i % 500 == 0:
                # remove all past checkpoints
                print(f"iterdir {list((ROOT_DIR / 'temp').iterdir())}")
                listremove = list(
                    filter(lambda x: "ckpt" in str(x), (ROOT_DIR / "temp").iterdir())
                )
                print(f"listremove {listremove}")
                for f in listremove:
                    os.remove(f)
                with open(
                    ROOT_DIR / "temp" / f"company_wikipedia_profiles_ckpt{i}", "w"
                ) as f:
                    dump(company_names, f)
        except requests.exceptions.ReadTimeout:
            continue
end = time()
print(f"len {len(dfnews.company_name.unique())} summaries took {end-start:.2f} seconds")
dfnews["company_context"] = dfnews["company_name"].apply(lambda x: company_names[x])
dfnews.to_parquet(ROOT_DIR / "temp" / "data_v5_company_context_done.parquet")
