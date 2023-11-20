# %%

### this pipelin uses pseudo labels instead of RET as a target.
### Why new pipeline ? because the amount of pseudo labels is so low that train val test split does not work
## why use a simple 0.6/0.4 train-val in this pipeline

# %%
import gc
import re
import sys
import itertools
import os
from typing import Dict, List, Any, Optional
from collections import defaultdict
import pandas as pd
import torch
from datasets import DatasetDict
import tokenizers
from transformers import Trainer
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers.integrations import NeptuneCallback
import transformers as tr
import neptune
import logging
import numpy as np
from itertools import chain
import ctypes
from pathlib import Path
import random
import json 

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

from utils.stats_utils import model_memory_used, count_parameters
from utils.neptune_utils import log_run
from utils.io_utils import write_log_to_file, load_peft_model_from_files
from models.load_model_v14 import load_model_pretrained
from dotenv import load_dotenv
from utils.pipeline_utils import is_notebook, set_handlers, clean_memory
load_dotenv(ROOT_DIR / "conf.env")

try:
    import flash_attn
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)



# %%
# arguments
@dataclass
class ModelArguments:
    model_name: Optional[str] = field(
        default="meta_llama/Llama-2-7b-hf"
    )  # ["PY007/TinyLlama-1.1B-step-50K-105b", ]
    lora_dim: int = field(default=16)
    lora_dropout: float = field(default=0.3)
    mlp_dropout: float = field(default=0.3)
    pooler_type_logits: str = field(default="avg")
    use_mlp: bool = field(default=True)
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "query", "key", "value"]
    )
    num_labels: int = field(default=2)


@dataclass
class DataArguments:
    filename_headlines: str = field(
        default="news_headlines_v5.2.parquet",
        metadata={"help": "Path to the training data."},
    )
    x_col: str = field(
        default="headline_no_ent",
        metadata={"help": "Name of the column containing the news headlines."},
    )
    y_col: str = field(
        default="RET_10D_pos",
        metadata={"help": "Name of the column containing the labels."},
    )
    num_contexts: int = field(
        default=5,
        metadata={"help": "Number of past headlines to use as context."},
    )


@dataclass
class TrainingArguments(tr.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    is_debug: bool = field(default=False)
    exp_name: str = field(default="llama7b_v14")


# %%
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
    ):
        """
        This dataset implements a dynamic context, where the context is the past"""
        self.y_col = y_col
        if y_cols_context:
            self.y_cols = y_cols_context
        else:
            self.y_cols = [y_col]
        self.debug = debug
        # reset indexes correctly
        if "cluster" in df.index.names:
            df.reset_index(level="cluster", drop=True, inplace=True)
        df = df.reset_index().set_index(["SYMBOL", "DATE"]).sort_index()
        df = df.sort_values(by=["cluster", "DATE", "SYMBOL"])
        assert self.y_col in df.columns
        self.x_col = x_col
        assert self.x_col in df.columns
        self.df = df[[x_col] + self.y_cols + ["cluster"]]
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
        mask1 = self.df.cluster == current_row.cluster
        mask2 = self.df.index.get_level_values(1) < maxdatelp
        if any(mask1 & mask2):
            mask = mask1 & mask2
        else:
            mask = mask2
        context = self.df.loc[mask].iloc[-self.num_contexts :][
            [self.x_col] + self.y_cols + ["cluster"]
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
        if self.debug:
            context_dict["DATE"] = self.df.iloc[idx].name[1]
            context_dict["cluster"] = self.df.iloc[idx]["cluster"]
            context_dict["SYMBOL"] = self.df.iloc[idx].name[0]
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
        if isinstance(idx, tuple):
            # add prompt
            if idx[1]:
                ret = self.__getitem__(idx[0])
                ret["prompt"] = self.prompts[idx[0]]
                return ret
            else:
                # dont add prompt
                idx = idx[0]
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
        return ret

    def __len__(self):
        return self.df.shape[0]



# %%
# test the dataset
def make_default_args(default_args:dict):
    default_args = [(f"--{k}", str(v)) for k, v in default_args.items()]
    default_args = list(itertools.chain(*default_args))
    return default_args

args = tr.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
if is_notebook():
    default_values = make_default_args(dict(json.loads(ROOT_DIR/"pipeline_confs/default_args.json")))
    model_args, data_args, training_args = args.parse_args_into_dataclasses(
        default_values
    )
else:
    model_args, data_args, training_args = args.parse_args_into_dataclasses()
model_args.device_map = "auto" if training_args.deepspeed is None else None


# %%


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
    newsdf = newsdf[[x_col, y_col, "cluster"]]

    # train val test split
    train_fraction = 0.6
    val_fraction = 0.2
    test_fraction = 0.2
    if "cluster" in newsdf.index.names:
        newsdf = newsdf.reset_index(level="cluster", drop=True)
    newsdf = newsdf.sort_values(by=["DATE", "cluster", "SYMBOL"])
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
    return dataset_full


# %%
data_folder = os.environ["PATH_DATA"]
dataset_full = load_dataset_pseudo_label(
    Path(data_folder) / data_args.filename_headlines,
    model_name=model_args.model_name,
    x_col=data_args.x_col,
    y_col=data_args.y_col,
    num_contexts=data_args.num_contexts,
    num_labels=model_args.num_labels,
    debug=training_args.is_debug,
    y_cols_context=[data_args.y_col],
)

# %%
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


collator_fn = DataCollatorCustom(tokenizer=dataset_full["train"].tokenizer)


# %%
model = load_model_pretrained(model_args, training_args, mlp_version=3)


def compute_metrics(pred):
    # accuracy
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = (labels == preds).mean()
    return {"accuracy": acc}

assert training_args.evaluation_strategy == "steps"
assert training_args.eval_steps == 25
assert len(dataset_full["eval"])>1e3
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_full["train"],
    eval_dataset=dataset_full["validation"],
    data_collator=collator_fn,
    compute_metrics=compute_metrics,
)


# %%
run = next(
    filter(lambda x: isinstance(x, NeptuneCallback), trainer.callback_handler.callbacks)
).run
len_dataloader_train = len(trainer.get_train_dataloader())
sys_id = int(re.search(r"LLM-(\d+)", run.get_attribute("sys/id").fetch()).group(1))
log_run(
    run,
    model,
    model_args.model_name,
    training_args,
    len_dataloader_train,
    HAS_FLASH_ATTN,
    dataset_full["train"],
    filename_headlines=data_args.filename_headlines,
)

os.makedirs(training_args.output_dir, exist_ok=True)
logger = set_handlers(logger, sys_id)  # creates a file handler.


# %%

# saw this in training scripts online, not sure if it's necessary
for name, module in trainer.model.named_modules():
    if "norm" in name:
        module = module.to(torch.bfloat16)  # might be better to use float32
    if "score" in name:
        module = module.to(torch.float32)  # might be better to use float32


# only train the classification layer, no lora
if model_args.lora_dim == 0:
    for name, param in trainer.model.named_parameters():
        if "score" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

trainable, total = count_parameters(trainer.model)
print(f"trainable {trainable}, total {total}, ratio {trainable/total:.3f}")


# %%
dataloader = torch.utils.data.DataLoader(
    dataset_full["train"], batch_size=2, shuffle=True, collate_fn=collator_fn
)
batch = next(itertools.islice(iter(dataloader), 1000, 1001))
for i, batch in enumerate(dataloader):
    question_mask = batch["question_mask"]
    for j in range(question_mask.shape[0]):
        assert any(
            question_mask[j] == 1
        ), f"question mask {question_mask[j]} does not contain any 1"

# %%

trainer.args.remove_unused_columns = False
dataloader2 = trainer.get_train_dataloader()
batch = next(itertools.islice(iter(dataloader2), 1000, 1001))
print(f"batch keys {batch.keys()}")

# %%


trainer.train()
from peft.utils.other import ModulesToSaveWrapper

# model is trained
# %%
if "test" in dataset_full:
    write_log_to_file(
        trainer.state.log_history,
        filepath=os.path.join(training_args.output_dir, "log_history.txt"),
    )
    test_preds = trainer.predict(test_dataset=dataset_full["test"])
    test_preds = test_preds.predictions
    test_preds_max = test_preds.argmax(axis=1)
    test_preds = test_preds_max
    print(f"test preds shape {test_preds.shape}")
    test_labels = dataset_full["test"]["labels"]
    accuracy_test = (test_preds_max == np.array(test_labels)).mean()

    num_tp = ((test_labels == 1) & (test_preds == 1)).sum()
    num_tn = ((test_labels == 0) & (test_preds == 0)).sum()
    num_fp = ((test_labels == 0) & (test_preds == 1)).sum()
    num_fn = ((test_labels == 1) & (test_preds == 0)).sum()
    confusion_mat = np.array([[num_tp, num_fp], [num_fn, num_tn]])
    print(f"                      True")
    print(f"                 1  |  0 ")
    print(f"      1  | {num_tp} | {num_fp}")
    print(f"Pred | 0|  {num_fn} | {num_tn}")

    from sklearn.metrics import f1_score, precision_score, recall_score

    f1score = f1_score(test_labels, test_preds)
    print(f"f1 score {f1score}")

    sys_id_int = int(re.findall(r"\d+", sys_id)[0])
    run = neptune.init_run(
        api_token=os.environ["NEPTUNE_API_TOKEN"],
        project=os.environ["NEPTUNE_PROJECT"],
        with_id=f"LLM-{sys_id_int}",
    )
    run["test/accuracy"] = accuracy_test
    print(f"test accuracy {accuracy_test}")

# %%
