# %%
import asyncio
import os
import pandas as pd
import numpy as np
from pathlib import Path
from itertools import chain
from typing import Optional

ROOT_DIR = next(
    filter(lambda s: "LLM" in s.name, Path.cwd().iterdir().__next__().parents), None
)
print(ROOT_DIR)
import configparser
import matplotlib.pyplot as plt

from dotenv import load_dotenv
load_dotenv(ROOT_DIR / "conf.env")
DATA_FOLDER = os.environ["PATH_DATA"]

print(DATA_FOLDER)
assert os.path.isdir(DATA_FOLDER)
assert os.path.isdir(ROOT_DIR)
import sys
import re
from tqdm import tqdm

sys.path.append(str(ROOT_DIR))
pd.set_option("display.max_colwidth", None)
import openai, random
from fake_openai import OpenAIFake
import json

from utils.timeout_function import TimeoutWrapper



# %%


def build_prompt_from_attr(prompt_attr: dict):
    prompt = "You are a financial analyst. For the following news write a return forecast. Answer using 'UP' or 'DOWN' or 'NOT RELEVANT'. Do not rewrite the returns for the example 1., 2., and 3. \n###\n"
    for i in range(len(prompt_attr["news"])):
        number = i + 1  # Start numbering at 1
        prompt += f"{number}.\nNews:\n{prompt_attr['news'][i]}\nResponse:\n{prompt_attr['label'][i] if prompt_attr['label'][i] else ''}\n###\n"
    return prompt


# building prompt


def build_all_prompts(
    df: pd.DataFrame,
    limit: int = 3,
    num_rows_to_generate: int = 10,
    x_col: str = "headline_no_ent_v2",
    label_col: str = "pseudo_label",
):
    """
    df is a single cluster
    """
    mapping_direction_to_int = {"UP": 1, "DOWN": 0, "NOT RELEVANT": 2}
    mapping_int_to_direction = {0: "DOWN", 1: "UP", 2: "NOT RELEVANT"}
    prompt_attr = {"news": [], "label": []}
    generated_for_index_all = []
    df_labels_notna = df.loc[df[label_col].notna() & (df[label_col] != -1)]
    sample_labels_notna = df_labels_notna.sample(3)

    for i in range(sample_labels_notna.shape[0]):
        headline = sample_labels_notna[x_col].iloc[i]
        label = sample_labels_notna[label_col].iloc[i]
        label_str = mapping_int_to_direction[label]
        prompt_attr["news"].append(headline)
        prompt_attr["label"].append(label_str)

    prompts_all = []
    # create prompt for 10 rows at a time, by adding to prompt_attr 10 more news and 10 empty labels
    for i in range(0, df.shape[0], num_rows_to_generate):
        generated_for_index = []
        prompt_attr_new = {
            "news": prompt_attr["news"].copy(),
            "label": prompt_attr["label"].copy(),
        }
        for j in range(num_rows_to_generate):
            idx = i + j
            row = df.iloc[idx]
            label = row[label_col]
            idx_in_df = row["index_range"]
            if (
                isinstance(label, str)
                or not np.isnan(label)
                or (float(label) >= 0.0 and float(label) <= 2.0)
            ):
                continue  # this label is already generated
            headline = row[x_col]
            prompt_attr_new["news"].append(headline)
            prompt_attr_new["label"].append("")
            generated_for_index.append(idx_in_df)
        if len(prompt_attr_new["news"]) <= len(prompt_attr["news"]):
            print(f"skipping prompt {i} because no new labels to generate")
            continue
        prompt = build_prompt_from_attr(prompt_attr_new)
        prompts_all.append(prompt)
        generated_for_index_all.append(generated_for_index)
        if len(prompts_all) >= limit:
            break
    return prompts_all, generated_for_index_all


def parse_content(content):
    # 3. UP
    # 4. NOT RELEVANT
    # 5. DOWN
    # 6. DOWN
    # 7. DOWN
    # 8. UP
    # 9. DOWN
    if not content:
        # case of a timeout, for example
        return None
    content = content.split("\n")
    content = [c for c in content if c != ""]
    content_final = []
    for k in range(len(content)):
        re_pattern = r"(UP|DOWN|NOT RELEVANT)"
        match = re.search(re_pattern, content[k])
        if match:
            match_result = match.group(1)
            content_final.append(match_result)

    return content_final


def insert_content_into_df(
    content_final: list[str],
    generated_for_index: list[int],
    df: pd.DataFrame,
    label_col: str = "pseudo_label",
):
    """
    content_final : ex.["UP", "DOWN", "NOT" "RELEVANT", "UP", "DOWN"]
    """
    mapping_direction_to_int = {"UP": 1, "DOWN": 0, "NOT RELEVANT": 2}
    try:
        assert len(content_final) == len(generated_for_index)
    except AssertionError as e:
        print(f"error in content generated by gpt")
        print(f"content final {content_final}")
        print(f"generated for index {generated_for_index}")
        return df

    # inserting labels into df
    for k in range(len(content_final)):
        index_to_label = generated_for_index[k]
        pseudolabel_idx = df.columns.get_loc(label_col)
        df.iloc[index_to_label, pseudolabel_idx] = mapping_direction_to_int.get(
            content_final[k], -1
        )
    return df


# %%


def get_api_responses_v1_timeout(
    prompts_all, use_fake: bool = False, timeout: int = 12
):
    # adding timeout to the openai api call

    contents = []
    for i, prompt in tqdm(enumerate(prompts_all), total=len(prompts_all)):
        if use_fake:
            sleep_time = random.choice([1, 7])
            client = OpenAIFake(sleep_time)
        else:
            client = openai.OpenAI()
        re_pattern = r"(\d+).\nNews"
        final_int = int(re.findall(re_pattern, prompt)[-1])
        function = client.chat.completions.create
        args = []

        kwargs = {
            "model": "gpt-3.5-turbo-1106",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 150,
            "stop": [f"{final_int+1}.", f" {final_int+1}."],
        }
        function = TimeoutWrapper(limit=timeout)(function)
        function(*args, **kwargs)
        while True:
            try:
                response = function.get_value()
                if response is not None:
                    break
            except TimeoutError:
                print(f"timeout error for prompt {i}")
                response = None
                break
        if response is None:
            content = ""
        else:
            content = response.choices[0].message.content
            for rp in range(2, len(content) - 1):
                re_pattern = r"(\d+)."
                match_re = re.search(re_pattern, content[rp - 2 : rp + 1])
                if match_re:
                    match_result = match_re.group(1)
                    if int(match_result) == final_int + 1:
                        content = content[: rp - 2]

        contents.append(content)
    return contents


# %%
def main_v3(
    df: pd.DataFrame, cluster: int, use_fake: bool, num_prompts_per_cluster: int = 3
):
    print(f"running pseudo labelling for cluster {cluster}, with use_fake {use_fake}")
    # only add labelling to one cluster
    if cluster not in df.cluster.unique():
        print(f"cluster {cluster} not in df. Returning the df as is.")
        return df
    subdf = df[df.cluster == cluster]
    prompts_all, generated_for_index_all = build_all_prompts(
        subdf, limit=num_prompts_per_cluster
    )
    if use_fake: 
        print(prompts_all[0])
    contents = get_api_responses_v1_timeout(
        prompts_all, use_fake=use_fake, timeout=3 if use_fake else 12
    )
    for i, content in enumerate(contents):
        content_final = parse_content(content)
        if content_final is None:
            content_final = ["" for _ in range(len(generated_for_index_all[i]))]
        df = insert_content_into_df(content_final, generated_for_index_all[i], df)
    return df


# %%


def pipeline_for_cluster(use_fake: bool = False, num_prompts_per_cluster: int = 3):
    # if you want to start over, use 1.1.parquet, otherwise use 1.2.parquet
    dfmanual = pd.read_parquet(
        ROOT_DIR / "data_preprocess/data_v6/temp_pseudo_labels_v1.3.parquet"
    )

    for cluster in range(1, 10):
        # add cluster column
        if "cluster" in dfmanual.index.names:
            dfmanual = dfmanual.reorder_levels(
                ["cluster", "DATE", "SYMBOL"]
            ).sort_index()
        else:
            dfmanual = (
                dfmanual.reset_index()
                .set_index(["cluster", "DATE", "SYMBOL"])
                .sort_index()
            )
        dfmanual["cluster"] = dfmanual.index.get_level_values("cluster")

        # add index col
        if "index_range" not in dfmanual.columns:
            dfmanual["index_range"] = pd.Series(
                range(dfmanual.shape[0]), index=dfmanual.index
            )
        print(
            f"number of labels total before inserting {dfmanual.loc[dfmanual.pseudo_label.notna(), :].shape[0]}"
        )
        # apply the psuedo labelling
        dfmanual = main_v3(
            dfmanual, cluster, use_fake, num_prompts_per_cluster=num_prompts_per_cluster        )
        print(
            f"number of labels total after inserting {dfmanual.loc[dfmanual.pseudo_label.notna(), :].shape[0]}"
        )

    if use_fake == False:
        dfmanual.to_parquet(
            ROOT_DIR / "data_preprocess/data_v6/temp_pseudo_labels_v1.3.parquet"
        )
        print(
            f"saving to {ROOT_DIR / 'data_preprocess/data_v6/temp_pseudo_labels_v1.3.parquet'}"
        )
        df_local = pd.read_parquet(
            Path(os.environ["PATH_DATA"]) / "temp_pseudo_labels_v1.3.parquet"
        )
        not_na_local = df_local.loc[df_local.pseudo_label.notna(), :].shape[0]
        notna_manual = dfmanual.loc[dfmanual.pseudo_label.notna(), :].shape[0]
        if notna_manual>not_na_local:
            print(f"notna_manual {notna_manual} > not_na_local {not_na_local}")
            dfmanual.to_parquet(
                Path(os.environ["PATH_DATA"]) / "temp_pseudo_labels_v1.3.parquet"
            )
            print(
                f"saving to {Path(os.environ['PATH_DATA']) / 'temp_pseudo_labels_v1.3.parquet'}"
            )

    else:
        dfmanual.to_parquet(
            ROOT_DIR / "data_preprocess/data_v6/temp_pseudo_labels_v1.2_fake.parquet"
        )
        print(
            f"saving to {ROOT_DIR / 'data_preprocess/data_v6/temp_pseudo_labels_v1.2_fake.parquet'}"
        )
    print(
        f"number of labels for this cluster after inserting {dfmanual.loc[dfmanual.pseudo_label.notna(), :].shape[0]}"
    )




if __name__ == "__main__":
    from fire import Fire
    Fire(pipeline_for_cluster)

# %%
