# %%
import asyncio
import os
import pandas as pd
import numpy as np
from pathlib import Path
from itertools import chain

ROOT_DIR = next(
    filter(lambda s: "LLM" in s.name, chain(Path(os.getcwd()).parents, [Path.cwd()]))
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
import re
from tqdm import tqdm

sys.path.append(str(ROOT_DIR))
pd.set_option("display.max_colwidth", None)
import openai, random
from fake_openai import OpenAIFake
import grequests, json

# %%


def chatgpt_labelling(
    df: pd.DataFrame,
    x_col: str,
    label_col: str,
    num_rows_to_generate: int = 10,
    use_fake: bool = False,
):
    # this df has one single cluster. We will generate labels for the rows that do not have a label yet
    if df.loc[df[label_col].notna()].shape[0] == df.shape[0]:
        print(f"all labels are already generated for this cluster")
        return df.copy()
    df = df.copy()
    assert (config.get("openai", "api_key")) is not None

    if use_fake:
        client = OpenAIFake()
    else:
        client = openai.OpenAI()

    num_rows_to_generate = 10
    mapping_direction_to_int = {"UP": 1, "DOWN": 0, "NOT RELEVANT": 2}
    mapping_int_to_direction = {0: "DOWN", 1: "UP", 2: "NOT RELEVANT"}
    df_labels_notna = df.loc[df[label_col].notna()]
    sample_labels_notna = df_labels_notna.sample(3)

    def build_prompt_from_attr(prompt_attr: dict):
        prompt = "You are a financial analyst. For the following news write a return forecast. Answer using 'UP' or 'DOWN' or 'NOT RELEVANT'. Do not rewrite the returns for the example 1., 2., and 3. Do not skip any question after 3.\n###\n"
        for i in range(len(prompt_attr["news"])):
            number = i + 1  # Start numbering at 1
            prompt += f"{number}.\nNews:\n{prompt_attr['news'][i]}\nResponse:\n{prompt_attr['label'][i] if prompt_attr['label'][i] else ''}\n###\n"
        return prompt

    # building prompt
    prompt_attr = {"news": [], "label": []}
    generated_for_index = []
    for i in range(sample_labels_notna.shape[0]):
        headline = sample_labels_notna[x_col].iloc[i]
        label = sample_labels_notna[label_col].iloc[i]
        label_str = mapping_int_to_direction[int(label)]
        prompt_attr["news"].append(headline)
        prompt_attr["label"].append(label_str)
    for j in range(df.shape[0]):
        row = df.iloc[j]
        label = row[label_col]
        if (
            isinstance(label, str)
            or not np.isnan(label)
            or (float(label) >= 0.0 and float(label) <= 2.0)
        ):
            continue  # this label is already generated
        headline = row[x_col]
        prompt_attr["news"].append(headline)
        prompt_attr["label"].append("")
        generated_for_index.append(j)
        if len(generated_for_index) >= num_rows_to_generate:
            break
    prompt = build_prompt_from_attr(prompt_attr)

    # gpt3.5 api call
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=200,
        top_p=1.0,
        temperature=0.5,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=["14.", " 14."],
    )

    def parse_content(content):
        # 3. UP
        # 4. NOT RELEVANT
        # 5. DOWN
        # 6. DOWN
        # 7. DOWN
        # 8. UP
        # 9. DOWN
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

    content = response.choices[0].message.content
    content_final = parse_content(content)
    assert len(content_final) == len(generated_for_index)

    # inserting labels into df
    for k in range(len(content_final)):
        index_to_label = generated_for_index[k]
        pseudolabel_idx = df.columns.get_loc(label_col)
        df.iloc[index_to_label, pseudolabel_idx] = mapping_direction_to_int[
            str(content_final[k])
        ]
    return df


# %%
async def apply_labelling_loop_asyncio(df: pd.DataFrame, use_fake: bool = False):
    from tqdm import tqdm

    progress_bar = tqdm(total=df.shape[0])

    # if cluster is both an index and a column, then reset_index
    not_na_shape = df.loc[df.pseudo_label.notna()].shape
    progress_bar.update(not_na_shape[0])
    for k in range(1):
        print(f"running loop {k}")
        if "cluster" in df.index.names:
            if "cluster" in df.columns:
                df = df.reset_index(level="cluster", drop=True)
            else:
                df = df.reset_index(level="cluster", drop=False)

        await apply_groupby(df, use_fake=use_fake)  # 10 api calls
        if "cluster" in df.index.names:
            df = df.reset_index(level="cluster", drop=True)

        progress_bar.update(100)
        progress_bar.refresh()

    # save dfmanual back
    # df.to_parquet(
    #     ROOT_DIR / "data_preprocess/data_v6/temp_pseudo_labels_v1.2.parquet"
    # )


# %%
def apply_labelling_loop(dfmanual, use_fake: bool = False):
    progress_bar = tqdm(total=dfmanual.shape[0])

    # if cluster is both an index and a column, then reset_index
    not_na_shape = dfmanual.loc[dfmanual.pseudo_label.notna()].shape
    progress_bar.update(not_na_shape[0])
    for k in range(10):
        if "cluster" in dfmanual.index.names:
            if "cluster" in dfmanual.columns:
                dfmanual = dfmanual.reset_index(level="cluster", drop=True)
            else:
                dfmanual = dfmanual.reset_index(level="cluster", drop=False)
        tqdm.pandas()
        groupby_result = dfmanual.groupby("cluster").progress_apply(
            lambda x: chatgpt_labelling(
                x,
                "headline_no_ent_v2",
                "pseudo_label",
                num_rows_to_generate=10,
                use_fake=use_fake,
            )
        )
        not_na_shape2 = groupby_result.loc[groupby_result.pseudo_label.notna()].shape
        diff_labeling = not_na_shape2[0] - not_na_shape[0]
        assert diff_labeling == 100
        not_na_shape = not_na_shape2
        if "cluster" in groupby_result.index.names:
            groupby_result = groupby_result.reset_index(level="cluster", drop=True)

        if groupby_result.index.equals(dfmanual.index):
            dfmanual["pseudo_label"] = groupby_result["pseudo_label"]

        progress_bar.update(diff_labeling)
        progress_bar.refresh()

    # save dfmanual back
    dfmanual.to_parquet(
        ROOT_DIR / "data_preprocess/data_v6/temp_pseudo_labels_v1.1.parquet"
    )


# %%


async def main1(use_fake: bool, use_asyncio: bool):
    shape_before = dfmanual.loc[dfmanual.pseudo_label.notna(), :].shape
    from time import time

    start = time()
    if use_asyncio:
        await apply_labelling_loop_asyncio(dfmanual, use_fake=use_fake)
    else:
        apply_labelling_loop(dfmanual, use_fake=use_fake)
    end = time()
    print(f"took {end-start} seconds")

    # get diff
    shape_after = dfmanual.loc[dfmanual.pseudo_label.notna(), :].shape
    diff = shape_after[0] - shape_before[0]
    print(f"generated {diff} new labels")


# %%


async def chatgpt_labelling_asyncio(
    df: pd.DataFrame,
    x_col: str,
    label_col: str,
    num_rows_to_generate: int = 10,
    use_fake: bool = False,
):
    # this df has one single cluster. We will generate labels for the rows that do not have a label yet
    print(f"starting to label for cluster {df.cluster.unique()}")
    df = df.copy()
    if df.loc[df[label_col].notna()].shape[0] == df.shape[0]:
        print(f"all labels are already generated for this cluster")
        return df
    # df = df.copy()
    assert (config.get("openai", "api_key")) is not None

    if use_fake:
        client = OpenAIFake()
    else:
        client = openai.OpenAI()

    num_rows_to_generate = 10
    mapping_direction_to_int = {"UP": 1, "DOWN": 0, "NOT RELEVANT": 2}
    mapping_int_to_direction = {0: "DOWN", 1: "UP", 2: "NOT RELEVANT"}
    df_labels_notna = df.loc[df[label_col].notna()]
    sample_labels_notna = df_labels_notna.sample(3)

    def build_prompt_from_attr(prompt_attr: dict):
        prompt = "You are a financial analyst. For the following news write a return forecast. Answer using 'UP' or 'DOWN' or 'NOT RELEVANT'. Do not rewrite the returns for the example 1., 2., and 3. \n###\n"
        for i in range(len(prompt_attr["news"])):
            number = i + 1  # Start numbering at 1
            prompt += f"{number}.\nNews:\n{prompt_attr['news'][i]}\nResponse:\n{prompt_attr['label'][i] if prompt_attr['label'][i] else ''}\n###\n"
        return prompt

    # building prompt
    prompt_attr = {"news": [], "label": []}
    generated_for_index = []
    for i in range(sample_labels_notna.shape[0]):
        headline = sample_labels_notna[x_col].iloc[i]
        label = sample_labels_notna[label_col].iloc[i]
        label_str = mapping_int_to_direction[int(label)]
        prompt_attr["news"].append(headline)
        prompt_attr["label"].append(label_str)
    for j in range(df.shape[0]):
        row = df.iloc[j]
        label = row[label_col]
        if (
            isinstance(label, str)
            or not np.isnan(label)
            or (float(label) >= 0.0 and float(label) <= 2.0)
        ):
            continue  # this label is already generated
        headline = row[x_col]
        prompt_attr["news"].append(headline)
        prompt_attr["label"].append("")
        generated_for_index.append(j)
        if len(generated_for_index) >= num_rows_to_generate:
            break
    prompt = build_prompt_from_attr(prompt_attr)

    # gpt3.5 api call
    response = await client.chat.completions.acreate(
        model="gpt-3.5-turbo-1106",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=200,
        top_p=1.0,
        temperature=0.5,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=["14.", " 14."],
    )
    print(f"received response for cluster {df.cluster.unique()}")

    def parse_content(content):
        # 3. UP
        # 4. NOT RELEVANT
        # 5. DOWN
        # 6. DOWN
        # 7. DOWN
        # 8. UP
        # 9. DOWN
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

    content = response.choices[0].message.content
    content_final = parse_content(content)
    try:
        assert len(content_final) == len(generated_for_index)
    except AssertionError as e:
        print(f"content {content}")
        print(f"content final {content_final}")
        print(f"generated for index {generated_for_index}")
        raise e

    # inserting labels into df
    for k in range(len(content_final)):
        index_to_label = generated_for_index[k]
        pseudolabel_idx = df.columns.get_loc(label_col)
        df.iloc[index_to_label, pseudolabel_idx] = mapping_direction_to_int[
            str(content_final[k])
        ]
    return df


async def apply_groupby(df: pd.DataFrame, use_fake: bool = False):
    tasks = []
    for cluster in df.cluster.unique():
        subdf = df[df.cluster == cluster]
        # result = await chatgpt_labelling_asyncio(subdf, "headline_no_ent_v2", "pseudo_label", num_rows_to_generate=10, use_fake = use_fake)
        # df.loc[df.cluster == cluster, "pseudo_label"] = result["pseudo_label"]
        tasks.append(
            chatgpt_labelling_asyncio(
                subdf,
                "headline_no_ent_v2",
                "pseudo_label",
                num_rows_to_generate=10,
                use_fake=use_fake,
            )
        )
    # reorder the tasks
    result_dfs = await asyncio.gather(*tasks)
    for i, cluster in enumerate(df.cluster.unique()):
        print(f"updating cluster {cluster}")
        df.loc[df.cluster == cluster, "pseudo_label"] = result_dfs[i]["pseudo_label"]


# %%  making the chatgpt async


# %%
import asyncio
import aiohttp


async def get(
    session: aiohttp.ClientSession, prompt: str, model_name: str = "gpt-3.5-turbo-1106"
) -> dict:
    openai_url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
    }
    data_dict = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You are a helpful financial assistant."},
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "max_tokens": 100,
        "stop": ["14.", " 14."],
    }
    data_json = json.dumps(data_dict)
    print(f"Requesting {openai_url}")
    resp = await session.request(
        "POST", url=openai_url, data=data_json, headers=headers
    )
    data = await resp.json()
    print(f"Received data for {openai_url}")
    return data


def get_api_responses(
    prompts: list[str], use_async: bool = True, model: str = "gpt-3.5-turbo-1106"
):
    openai_url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
    }
    datas = list()
    for prompt in prompts:
        data_dict = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a helpful financial assistant."},
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "max_tokens": 100,
            "stop": ["14.", " 14."],
        }
        data_json = json.dumps(data_dict)
        datas.append(data_json)
    if use_async:
        rs = [grequests.post(openai_url, headers=headers, data=data) for data in datas]
        print(f"calling {len(rs)} api calls concurrently...")
        responses = grequests.map(rs)
    else:
        import requests

        responses = []
        for data in datas:
            response = requests.post(openai_url, headers=headers, data=data)
            print(f"received response {response}")
            responses.append(response)
    print(f"received {len(responses)} responses")
    response_contents = [
        response.json()["choices"][0]["message"]["content"] for response in responses
    ]
    return response_contents


async def get_api_responses_v2(prompts_all):
    # Asynchronous context manager.  Prefer this rather
    # than using a different session for each GET request
    async with aiohttp.ClientSession() as session:
        tasks = []
        for prompt in prompts_all:
            tasks.append(get(session=session, prompt=prompt))
        # asyncio.gather() will wait on the entire task set to be
        # completed.  If you want to process results greedily as they come in,
        # loop over asyncio.as_completed()
        htmls = await asyncio.gather(*tasks, return_exceptions=True)
        print(f"finished receiving {len(htmls)} responses")
        print(f"html 0 {htmls[0]}")


async def get_api_responses_v3(prompts_all):
    async with aiohttp.ClientSession() as session:
        tasks = [get(session=session, prompt=prompt) for prompt in prompts_all]
        htmls = []
        for task in asyncio.as_completed(tasks):
            html = await task
            print(f"received response {html}")
            htmls.append(html.json()["choices"][0]["message"]["content"])
        print(f"finished receiving {len(htmls)} responses")
        print(f"html 0 {htmls[0]}")
        return htmls


# %%
