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

ROOT_DIR = next(filter(
    lambda p: "LLM_project" in p.name, Path.cwd().iterdir().__next__().parents), None) 
print(f"ROOT_DIR {ROOT_DIR}")
sys.path.append(str(ROOT_DIR))
from dotenv import load_dotenv
load_dotenv(ROOT_DIR / "conf.env")

from utils.stats_utils import model_memory_used, count_parameters
from utils.neptune_utils import log_run
from utils.io_utils import write_log_to_file
from models.load_model_v14 import load_model_pretrained
from utils.pipeline_utils import is_notebook, set_handlers
from data_modules import DataCollatorCustom,DatasetBis, load_dataset_pseudo_label
from arg_classes import ModelArguments, DataArguments, TrainingArguments
try:
    import flash_attn
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)



# %% making the argument classes
args = tr.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
def make_default_args(default_args:dict):
    default_args = [(f"--{k}", str(v)) for k, v in default_args.items()]
    default_args = list(itertools.chain(*default_args))
    return default_args
if is_notebook():
    # fetch the default values in default_args.json
    default_args_path = ROOT_DIR/"pipeline_confs/default_args.json"
    assert default_args_path.exists()
    with open(default_args_path, "r") as fp:
        default_args_dict = json.load(fp)
    default_args = make_default_args(default_args_dict)
    model_args, data_args, training_args = args.parse_args_into_dataclasses(
        default_args
    )
else:
    #use the command line arguments
    model_args, data_args, training_args = args.parse_args_into_dataclasses()
model_args.device_map = "auto" if training_args.deepspeed is None else None




# %%
dataset_full = load_dataset_pseudo_label(
    Path(os.environ["PATH_DATA"]) / data_args.filename_headlines,
    model_name=model_args.model_name,
    x_col=data_args.x_col,
    y_col=data_args.y_col,
    num_contexts=data_args.num_contexts,
    num_labels=model_args.num_labels,
    debug=training_args.is_debug,
    y_cols_context=[data_args.y_col],
)


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
assert len(dataset_full["validation"])>1e3
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
