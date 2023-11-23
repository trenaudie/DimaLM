# %%

### this pipelin uses pseudo labels instead of RET as a target.
### Why new pipeline ? because the amount of pseudo labels is so low that train val test split does not work
## why use a simple 0.6/0.4 train-val in this pipeline

# %%
import sys
import os
from typing import Dict, List, Any
from collections import defaultdict
import pandas as pd
import torch
from datasets import DatasetDict

from tqdm import tqdm
from transformers import AutoTokenizer
import transformers as tr
import logging
import numpy as np
from itertools import chain
from pathlib import Path
import random
import json 
import itertools

ROOT_DIR = next(
    filter(
        lambda s: "LLM" in s.name, chain(Path().absolute().parents, [Path(os.getcwd())])
    ),
    None,
)
print(f"ROOT_DIR {ROOT_DIR}")
sys.path.append(str(ROOT_DIR))
from dotenv import load_dotenv
load_dotenv(ROOT_DIR / "conf.env")

try:
    import flash_attn
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)




class DatasetBis(torch.utils.data.Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        model_name: str,
        y_col: str,
        x_col: str,
        num_contexts: int = 5,
        debug: bool = True,
        y_cols_context: List[str] = [],
       use_context:bool = True,
    ):
        """
        This dataset implements a dynamic context, where the context is the past"""
        self.y_col = y_col
        if y_cols_context:
            self.y_cols = y_cols_context
        else:
            self.y_cols = [y_col]
        self.debug = debug
        self.keep_cols = [x_col] + self.y_cols
        if "cluster" in df.index.names:
            df.reset_index(level="cluster", drop=True, inplace=True)
        if "cluster" in df.columns:
            self.keep_cols += ["cluster"]
            self.use_cluster = True
        else:
            self.use_cluster = False
        df = df.reset_index().set_index(["SYMBOL", "DATE"])
        df = df.sort_values(by=[ "DATE", "SYMBOL"])
        assert self.y_col in df.columns
        self.x_col = x_col
        assert self.x_col in df.columns
        self.df = df[self.keep_cols]
        self.num_contexts = num_contexts
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.prompt_examples = []
        data_dict = self.preprocess_rows()
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.prompts = data_dict["prompt"]
        self.question_mask = data_dict["question_mask"]

    def getrow(self, idx):
        current_row = self.df.iloc[idx]
        date = current_row.name[1]
        maxdatelp = date - pd.Timedelta(days=5)
        if self.use_cluster:
            mask1 = self.df.cluster == current_row.cluster
        else :
            mask1 = True 
        mask2 = self.df.index.get_level_values(1) < maxdatelp
        if any(mask1 & mask2):
            mask = mask1 & mask2
        else:
            mask = mask2
        context = self.df.loc[mask].iloc[-self.num_contexts :][
           self.keep_cols
        ]
        context_dict = {}
        for ycol in self.y_cols:
            context_dict[f"past_{ycol}"] = (
                context[ycol].tolist() if context.shape[0] > 0 else []
            )
        context_dict[f"past_{self.x_col}"] = (
            context[self.x_col].tolist() if context.shape[0] > 0 else []
        )
        context_dict[self.x_col] = self.df.iloc[idx][self.x_col]
        context_dict[self.y_col] = self.df.iloc[idx][self.y_col]
        return context_dict

    def build_prompt(self, row_data: dict) -> str:
        """
        Generate a prompt for predicting the return of a company based on past headlines and returns.

        Args:
            row_data (dict): A dictionary containing relevant data for generating the prompt.

        Returns:
            str: A formatted prompt string for predicting the return (UP or DOWN) for the current news.

        Example of 'row_data' structure:
        {
            'past_RET_5D_pos': [1, 1, 1, 1, 1],
            'past_RET_10D_pos': [1, 1, 1, 1, 0],
            'past_headline_no_ent_v2': [
                'company added to Focus List at Howard Weil  (timing uncertain)',
                'company says President and CEO Claiborne P. Deming will retire, effective 31-Dec-08',
                'company promotes Simon Hemus to president and COO',
                'company added to Focus List at JPMorgan',
                'company chooses not to participate in Capital Purchase Program'
            ],
            'headline_no_ent_v2': 'company names Jan Keltjens as president and CEO effective 1-Mar',
            'RET_10D_pos': 1,
            'DATE': Timestamp('2009-01-22 00:00:00'),
            'cluster': 5,
            'SYMBOL': 'B06HD3-R'
        }

        Returns a prompt string in the following format:
        "Given the prediction labels 'up' and 'down', read the given list of pairs of news headlines and past returns for the company
        and predict the return for the current news. Answer in the format UP or DOWN

        News: [past_headline_1]
        Return: [average_return_for_past_headlines]

        News: [past_headline_2]
        Return: [average_return_for_past_headlines]

        ...

        News: [current_headline]
        Return: "
        """
        mapping_return_to_direction = {0: "DOWN", 1: "UP", 2: "NOT RELEVANT"}
        # prompt = open("myprompt.txt","r").read()
        prompt = "Given the prediction labels 'up' and 'down', read the given list of pairs of news headlines and past returns for the company and predict the return for the current news. Answer in the forwat UP or DOWN\n"
        for past_idx in range(len(row_data[f"past_{self.x_col}"])):
            past_news = row_data[f"past_{self.x_col}"][past_idx]  #
            prompt += f"News: {past_news}\n"
            y_cols = [f"past_{y_col}" for y_col in self.y_cols]
            y_cols = [y_col for y_col in y_cols if y_col in row_data.keys()]
            past_return = np.mean([row_data[col][past_idx] for col in y_cols])
            past_return = int(past_return >= 0.5)
            past_return = mapping_return_to_direction[past_return]
            prompt += f"Return: {past_return}\n\n"
        prompt += f"News: {row_data[self.x_col]}\n"
        prompt += f"Return: "
        return prompt

    def preprocess_rows(self) -> Dict[str, List[Any]]:
        # return {"input_ids": [...], "labels": [...]}
        random_example_idx = random.sample(range(self.df.shape[0]), 10)
        data_dict = defaultdict(list)
        for i in tqdm(range(0, self.df.shape[0])):
            row = self.getrow(i)
            prompt = self.build_prompt(row)
            # extract the last news and return
            prompt_question = prompt.split("\n")[-3:]
            prompt_question = [line for line in prompt_question if len(line) > 5]
            prompt_question = "\n".join(prompt_question)

            # log some info
            if i in random_example_idx:
                self.prompt_examples.append(prompt)
            if i % (self.df.shape[0] // 5) == 0:
                print(f"idx {i}, prompt {prompt}")

            # tokenize both the total prompt and the question
            tokenized = self.tokenizer(
                prompt, padding=False, truncation=False, return_tensors="pt"
            )
            tokenizer_question = self.tokenizer(
                prompt_question, padding=False, truncation=False, return_tensors="pt"
            )
            input_ids, input_ids_question = (
                tokenized["input_ids"],
                tokenizer_question["input_ids"],
            )
            lenq = len(input_ids_question[0])  # length of the question encoding
            assert lenq > 5
            question_mask = torch.zeros(input_ids.shape[1], dtype=torch.int)
            question_mask[-lenq:] = 1
            assert any(question_mask == 1)

            data_dict["input_ids"].append(tokenized["input_ids"])
            data_dict["labels"].append(row[self.y_col])
            data_dict["prompt"].append(prompt)
            data_dict["question_mask"].append(question_mask)

        return data_dict

    def __getitem__(self, idx):
        ret = None
        if isinstance(idx, int):
            ret = {
                "input_ids": self.input_ids[idx][0],
                "labels": self.labels[idx],
                "question_mask": self.question_mask[idx],
            }
        elif isinstance(idx, list):
            ret = {
                "input_ids": list(map(lambda x: self.input_ids[x][0], idx)),
                "labels": list(map(lambda x: self.labels[x], idx)),
                "question_mask": list(map(lambda x: self.question_mask[x], idx)),
            }
            if ret is None: 
                raise NotImplementedError
        return ret

    def __len__(self):
        return self.df.shape[0]



def load_dataset_pseudo_label(
    filename_headlines: str,
    model_name: str,
    x_col: str,
    y_col: str,
    num_contexts: int,
    num_labels: int,
    debug: bool = False,
    y_cols_context: List[str] = [],
):
    print(f"reading from path {filename_headlines}")
    newsdf = pd.read_parquet(filename_headlines, engine="pyarrow")
    assert y_col in newsdf.columns
    assert x_col in newsdf.columns
    newsdf = newsdf.loc[(newsdf[y_col].notna()) & (newsdf[y_col] >= 0)]
    newsdf[y_col] = newsdf[y_col].astype(int)
    try:
        assert num_labels == newsdf[y_col].nunique()
    except AssertionError:
        print(f"unique values {newsdf[y_col].unique()}")
        raise AssertionError
    cols_keep = [x_col, y_col]
    cols_keep += ["cluster"] if "cluster" in newsdf.columns else []
    newsdf = newsdf[cols_keep]

    # train val test split
    train_fraction = 0.6
    val_fraction = 0.2
    if "cluster" in newsdf.index.names:
        newsdf = newsdf.reset_index(level="cluster", drop=True)

    newsdf = newsdf.sort_values(by=["DATE", "SYMBOL"])
    if debug:
        if newsdf.shape[0] > 3e4:
            symbols = newsdf.index.get_level_values("SYMBOL").unique()
            symbols_tenth = random.sample(list(symbols), int(len(symbols) * 0.01))
            newsdf = newsdf.loc[newsdf.index.get_level_values("SYMBOL").isin(symbols_tenth)]
    news_df_train = newsdf.iloc[: int(len(newsdf) * train_fraction)]
    max_date_train = news_df_train.index.get_level_values("DATE").max()
    news_df_val = newsdf.iloc[
        int(len(newsdf) * train_fraction) : int(
            len(newsdf) * (train_fraction + val_fraction)
        )
    ]
    news_df_test = newsdf.iloc[int(len(newsdf) * (train_fraction + val_fraction)) :]
    news_df_val = news_df_val.loc[
        news_df_val.index.get_level_values("DATE") > max_date_train + pd.Timedelta("7d")
    ]
    max_date_val = news_df_val.index.get_level_values("DATE").max()
    news_df_test = news_df_test.loc[
        news_df_test.index.get_level_values("DATE") > max_date_val + pd.Timedelta("7d")
    ]
    print(f"min date train {news_df_train.index.get_level_values('DATE').min()}")
    print(f"max date train {news_df_train.index.get_level_values('DATE').max()}")
    print(f"min date val {news_df_val.index.get_level_values('DATE').min()}")
    print(f"max date val {news_df_val.index.get_level_values('DATE').max()}")
    print(f"min date test {news_df_test.index.get_level_values('DATE').min()}")
    print(f"max date test {news_df_test.index.get_level_values('DATE').max()}")

    print(
        f"shapes train {news_df_train.shape}, val {news_df_val.shape} test {news_df_test.shape}"
    )
    assert news_df_train.shape[0] > 0, "train df empty"
    assert news_df_val.shape[0] > 0, "val df empty"
    assert news_df_test.shape[0] > 0, "test df empty"
    assert x_col in news_df_train.columns, f"{x_col} not in columns"

    dataset_train = DatasetBis(
        news_df_train,
        model_name=model_name,
        y_col=y_col,
        num_contexts=num_contexts,
        x_col=x_col,
        debug=debug,
        y_cols_context=y_cols_context,
    )
    dataset_val = DatasetBis(
        news_df_val,
        model_name=model_name,
        y_col=y_col,
        num_contexts=num_contexts,
        x_col=x_col,
        debug=debug,
        y_cols_context=y_cols_context,
    )
    dataset_test = DatasetBis(
        news_df_test,
        model_name=model_name,
        y_col=y_col,
        num_contexts=num_contexts,
        x_col=x_col,
        debug=debug,
        y_cols_context=y_cols_context,
    )

    dataset_full = DatasetDict(
        {
            "train": dataset_train,
            "validation": dataset_val,
            "test": dataset_test,
        }
    )
    assert len(dataset_full["train"]) > 10
    assert len(dataset_full["validation"]) > 10
    assert len(dataset_full["test"]) > 10
    return dataset_full


@dataclass
class DataCollatorCustom:
    tokenizer: tr.PreTrainedTokenizerBase

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        keys_features = [
            key
            for key in features[0].keys()
            if key in ("input_ids", "labels", "question_mask")
        ]
        features_listed = tuple(
            [instance[key] for instance in features] for key in keys_features
        )
        input_ids, labels = features_listed[0], features_listed[1]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.tensor(labels)
        ret = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        if "question_mask" in keys_features:
            question_mask = features_listed[2]
            question_mask = torch.nn.utils.rnn.pad_sequence(
                question_mask, batch_first=True, padding_value=0
            )
            ret["question_mask"] = question_mask
        return ret



