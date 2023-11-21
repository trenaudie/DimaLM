# %%
import os
import pandas as pd
import numpy as np
from pathlib import Path
from itertools import chain
import re
from collections import defaultdict
from typing import Dict, List, Any

ROOT_DIR = next(
    filter(
        lambda s: "LLM" in s.name, chain(Path(os.getcwd()).parents, [Path(os.getcwd())])
    ),
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
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
from itertools import islice
from transformers import DataCollatorWithPadding
import gc
import ctypes

sys.path.append(str(ROOT_DIR))


def clean_memory():
    gc.collect()
    ctypes.CDLL("libc.so.6").malloc_trim(0)
    torch.cuda.empty_cache()


# %%
def gen_df(df: pd.DataFrame, xcols: list[str]):
    for i in range(0, df.shape[0], 10):
        yield df.iloc[i : i + 10][xcols]


# %%
import re
import unicodedata
from tqdm import tqdm
import spacy
from collections import Counter
import re


# %% [markdown]
# ### Create company name list (a better company_name column)

# %%

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


def apply_get_company_name(df: pd.DataFrame):
    tqdm.pandas()
    mapping_symbol_to_name = df.groupby("SYMBOL").progress_apply(
        lambda x: get_company_names(x)[0]
    )
    df["company_name_no_ent_v2"] = df.index.get_level_values(0).map(
        lambda x: mapping_symbol_to_name.get(x, None)
    )
    df["company_name_no_ent_v2"].value_counts(dropna=False)
    df = df.rename(columns={"company_name_no_ent_v2": "company_name_list"})
    df.to_parquet(DATA_FOLDER / "news_headlines_v5.2.parquet")


# %%


def replace_headline(headline, company_name_list):
    def remove_accents(input_str):
        nfkd_form = unicodedata.normalize("NFKD", input_str)
        only_ascii = nfkd_form.encode("ASCII", "ignore")
        return only_ascii.decode("utf-8")

    # Normalize the company names and the headline for matching
    company_name_list_normalized = [remove_accents(name) for name in company_name_list]
    headline_normalized = remove_accents(headline)

    # Prepare the pattern with sorted names (longest to shortest to prevent partial replacement) and normalize it
    pattern = r"|".join(sorted(company_name_list_normalized, key=len, reverse=True))

    # Initialize a variable to store the success of the replacement
    replacement_success = True

    # Replace the pattern in the normalized headline
    try:
        headline_replaced = re.sub(
            pattern, "company", headline_normalized, flags=re.IGNORECASE
        )
    except re.error as e:
        print(f"Replacement error for headline '{headline_normalized}': {e}")
        headline_replaced = headline_normalized
        replacement_success = False

    # Return both the replaced headline and the success flag
    return headline_replaced, replacement_success


def apply_replace(df: pd.DataFrame):
    tqdm.pandas()
    results = df.progress_apply(
        lambda x: replace_headline(x["headline"], x["company_name_list"]), axis=1
    )
    df["headline_no_ent_v2"], df["headline_replaced_success_v2"] = zip(*results)
    print(f"value counts : {df.headline_replaced_success_v2.value_counts()}")
    df.to_parquet(DATA_FOLDER / "news_headlines_v5.2.parquet")


# %% [markdown]
# # Clustering

# %%


# Load model from HuggingFace Hub


# %%


def make_df_embeddings(
    df: pd.DataFrame,
    model_name: str = "BAAI/bge-large-zh-v1.5",
    headline_col: str = "headline_no_ent_v2",
    batchsize: int = 32,
    debug=True,
    device: str = "cuda:0",
    output_file="temp_df_embeds_v2.parquet",
):
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-large-zh-v1.5")
    model = AutoModel.from_pretrained("BAAI/bge-large-zh-v1.5")
    model.eval()
    torch.cuda.empty_cache()
    model = AutoModel.from_pretrained(model_name, device_map=device)
    model = model.eval()
    collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")
    sentence_embeddings_all = []
    if debug and df.shape[0] > 1e5:
        DEBUG_STOP = 10000
    else:
        DEBUG_STOP = df.shape[0]
    print(f"adding embeddings for {DEBUG_STOP} rows")
    df = df.iloc[:DEBUG_STOP]
    for k in tqdm(range(0, df.shape[0], batchsize)):
        try:
            sentences = df[headline_col].iloc[k : k + batchsize].tolist()
            encoded_input = list(
                tokenizer(sentence, padding=True, truncation=False, return_tensors=None)
                for sentence in sentences
            )
            collated_input = collator(encoded_input)

            with torch.no_grad():
                encoder_input = {k: v.to(device) for k, v in collated_input.items()}
                model_output = model(**encoder_input)
                # Perform pooling. In this case, cls pooling.
                sentence_embeddings = model_output[0][:, 0]
            # normalize embeddings
            sentence_embeddings = torch.nn.functional.normalize(
                sentence_embeddings, p=2, dim=1
            )
            # apply these sentence embeddings to the dataframe (shape (32,1024))
            sentence_embeddings_all.append(sentence_embeddings)
            clean_memory()
        except Exception as e:
            print("error at ", k)
            raise e
    sentence_embeddings_all = torch.cat(sentence_embeddings_all, dim=0)
    df = df.assign(embeddings=sentence_embeddings_all.cpu().numpy().tolist())
    df.to_parquet(output_file)
    return df


torch.cuda.empty_cache()
# djnews= make_df_embeddings(
#     djnews,
#     debug=False,
#     batchsize=16,
#     output_file="temp_df_embeds_v2.parquet",
# )
# %%


# %%
def create_clusters(df0: pd.DataFrame, num_clusters: int = 10):
    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=num_clusters, random_state=0, verbose=0).fit(
        df0["embeddings"].tolist()
    )
    df0["cluster"] = kmeans.labels_
    df0.to_parquet("temp_df_clusters_v3.parquet")
    return df0


# df0 = create_clusters(djnews)


# %% [markdown]
# adding past headline/return context to the clusters


# %%
df0 = pd.read_parquet(ROOT_DIR / "data_preprocess/data_v6/temp_df_clusters_v3.parquet")
assert "cluster" in df0.columns
df0.shape

# %%
gen = gen_df(df0.loc[df0["cluster"] == 0], ["headline_no_ent", "RET_5D_pos"])


# %%
next(gen)


# %% [markdown]
# ### Ideas for using the clusters
#
# How to best build the context window using similar lines
#
#
# 1. Naive approach :
# take the three closest (in the past) within the cluster.
# But the clusters are not balanced...
#
#
# 2. Topk clusters :
#
# for every row, dont just take headlines of the same cluster. Take the top5 headlines in the cluster (in terms of similarity score)
#
# 3. GPT4 pseudo labels of the clusters

# %%
# sliding window approach

# %%
# create "last3" context w/ cluster


# %%
# same thing, but using groupby cluster


def apply_sliding_context(df: pd.DataFrame, num_contexts: int = 5):
    lp = 0
    rp = 0
    # check that the df is sorted by date, and that there is only one cluster
    date = df.index.get_level_values(1)  # date
    assert date.is_monotonic_increasing
    assert df.cluster.nunique() == 1
    from collections import defaultdict

    makeNoneList = lambda: [None for _ in range(num_contexts)]
    context_list = [defaultdict(makeNoneList) for _ in range(df.shape[0])]
    for rp in tqdm(range(df.shape[0])):
        lp = max(0, rp - num_contexts)
        past_headlines = df.iloc[lp:rp]["headline_no_ent_v2"].astype(str).tolist()
        past_rets10d = df.iloc[lp:rp]["RET_10D_pos"].tolist()
        past_rets5d = df.iloc[lp:rp]["RET_5D_pos"].tolist()
        for k in range(rp - lp):
            context_list[rp]["ret10d"][k] = past_rets10d[k]
            context_list[rp]["ret5d"][k] = past_rets5d[k]
            context_list[rp]["news"][k] = past_headlines[k]
            context_list[rp]["news_current"] = df.iloc[rp]["headline"]

    print(f"for day 3 : {context_list[3]['news']}")
    context_df = pd.DataFrame(context_list, index=df.index)

    return context_df


# %%
df0.columns
df0 = df0.drop(columns=["return_context", "news_context"])

# %%
# apply sliding context but shift by 21d for the context


def apply_sliding_window_shifted(
    dfclustero: pd.DataFrame,
    shift_days: int = 21,
    num_contexts: int = 5,
    debug: bool = False,
):
    steps = 1 if not debug else 100
    # context dict is a default dict with lists
    from collections import defaultdict

    context_dict = defaultdict(list)
    for rp in tqdm(range(0, dfclustero.shape[0], steps)):
        indexrp = dfclustero.index[rp]
        daterp = indexrp[1]
        maxdatelp = daterp - pd.Timedelta(days=shift_days)
        context = dfclustero.loc[dfclustero.index.get_level_values(1) < maxdatelp].iloc[
            -num_contexts:
        ]
        # context_dict = {
        #     "ret10d": context["RET_10D_pos"].tolist(),
        #     "ret5d": context["RET_5D_pos"].tolist(),
        #     "news": context["headline_no_ent_v2"].tolist(),
        #     "news_current": dfclustero.iloc[rp]["headline"],
        # }

        for lag in range(1, num_contexts):
            context_dict[f"ret10d_lag{lag}"].append(context["ret10d"][-lag])
            context_dict[f"ret5d_lag{lag}"].append(context["ret5d"][-lag])
            context_dict[f"news_lag{lag}"].append(context["news"][-lag])

        context_dict["news_current"].append(dfclustero.iloc[rp]["headline"])
    assert len(context_dict["news_current"]) == dfclustero.shape[0]
    contexts_df_all_dates = pd.DataFrame(context_dict, index=dfclustero.index)
    return contexts_df_all_dates


dfclustero = df0.loc[df0["cluster"] == 0].sort_values(by=["DATE"])
# contexts_df = apply_sliding_window_shifted(dfclustero, debug=True)

# %%
df0 = df0.sort_values(by=["cluster", "DATE", "SYMBOL"])

# %%
from tqdm import tqdm

df0.head(5)

# %%
tqdm.pandas()
cluster_context_df = df0.groupby("cluster").progress_apply(
    apply_sliding_context, num_contexts=5
)
cluster_context_df = cluster_context_df.rename(
    columns={
        "ret10d": "ret10d_context",
        "ret5d": "ret5d_context",
        "news": "news_cluster_last3",
    }
)
cluster_context_df.columns

# %%
print(df0.shape, cluster_context_df.shape)
df1 = df0.merge(
    cluster_context_df[
        ["ret10d_context", "ret5d_context", "news_cluster_last3", "news_current"]
    ],
    left_on=["SYMBOL", "DATE", "headline"],
    right_on=["SYMBOL", "DATE", "news_current"],
    how="inner",
).drop(columns=["news_current"])
print(df1.shape)

# %%
cluster_context_df.head(10)
cluster_context_df.groupby(["SYMBOL", "DATE"]).size().sort_values(ascending=False).head(
    10
)


# %%
# check for duplicates in the SYMBOL DATE pair
df0.groupby(["SYMBOL", "DATE"]).size().sort_values(ascending=False).head(10)
# checking for PRLW4H-R  2018-11-08
symbol = "PRLW4H-R"
date = "2018-11-08"
df0.loc[
    (df0.index.get_level_values(0) == symbol) & (df0.index.get_level_values(1) == date)
]
# %%

df1.to_parquet("temp_df_v1.3.parquet")

# %%
