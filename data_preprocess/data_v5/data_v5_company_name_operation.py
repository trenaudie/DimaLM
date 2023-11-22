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
print(
    list(
        filter(
            lambda s: "LLM" in s.name,
            chain(Path().absolute().parents, [Path(os.getcwd())]),
        )
    )
)
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
from pathlib import Path
from tqdm import tqdm
import spacy
from collections import Counter
from utils.io_utils import write_headlines_to_file

nlp = spacy.load("en_core_web_sm")


def get_company_names(symboldf):
    full_text_symbol = ". ".join(symboldf["headline"].values)

    # NER and compound extraction
    compounds = []
    doc = nlp(full_text_symbol)
    for chunk in doc.noun_chunks:
        compounds.append(chunk.text)

    # Extract compound tokens and count the most common tokens
    common_names = [
        x[0] for x in filter(lambda x: len(x[0]) > 2, Counter(compounds).most_common(3))
    ]

    def is_potential_name(entry):
        # Check if it starts with an uppercase letter
        if not entry[0].isupper():
            return False
        # Filter out entries with reserved words - not implemented
        reserved_words = ["FactSet", "Reuters"]
        if any([word in entry for word in reserved_words]):
            return False
        # Check if the entry length is reasonable (e.g., between 2 to 5 words)
        if len(re.findall(r"\w+", entry)) > 5:
            return False
        return True

    # Detect abbreviations of names
    pattern = r"\b[A-Z]{2,}\b"
    is_abbrev = lambda x: re.match(pattern, x) is not None

    # Create a copy of common_names to iterate over without modification issues
    names_to_check = [entry for entry in common_names if is_potential_name(entry)]
    if not names_to_check:
        return [], common_names

    # if main name is lower case
    names_to_keep = [names_to_check[0]]
    for name in names_to_check:
        for name2 in names_to_check:
            if name2 != name:
                if name[0] == name2[0] and (is_abbrev(name) or is_abbrev(name2)):
                    names_to_keep.append(name)
                    names_to_keep.append(name2)
                    continue
                elif name2 in name or name in name2:
                    names_to_keep.append(name)
                    names_to_keep.append(name2)
                    continue
    names_to_keep = list(set(names_to_keep))
    # if one of the names is inside of another in the top 3, take it as well

    return list(set(names_to_keep)), common_names


if __name__ == "__main__":
    dfnews = pd.read_parquet(ROOT_DIR / "temp/data_v5_company_name.parquet")
    tqdm.pandas()
    mapping = dfnews.groupby("SYMBOL").progress_apply(
        get_company_name
    )  # each symbol gets a mapping
    dfnews["company_name"] = dfnews.index.get_level_values(0).map(mapping)
    # save the temp to temp/data_v5_company_name.parquet
    os.remove(ROOT_DIR / "temp/data_v5_company_name.parquet")
    dfnews.to_parquet(ROOT_DIR / "temp/data_v5_company_name_done.parquet")
    write_headlines_to_file(
        dfnews=dfnews,
        x_col="headline",
        add_cols=["company_name"],
        filepath=ROOT_DIR / "data_preprocess" / "news_headlines_v5_company_name.txt",
        limit=1000000,
        step=100,
        keep_date=False,
    )
