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
import torch
from transformers import Trainer
from transformers.integrations import NeptuneCallback
from transformers import AutoModelForCausalLM
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers import AutoConfig
import transformers as tr
import neptune
import logging
import numpy as np
from pathlib import Path
ROOT_DIR = next(filter(
    lambda p: "LLM_project" in p.name, Path.cwd().iterdir().__next__().parents), None) 
print(f"ROOT_DIR {ROOT_DIR}")
sys.path.append(str(ROOT_DIR))
from dotenv import load_dotenv
load_dotenv(ROOT_DIR / "conf.env")


from utils.stats_utils import model_memory_used, count_parameters
from utils.neptune_utils import log_run
from utils.io_utils import write_log_to_file
from utils.pipeline_utils import is_notebook, set_handlers
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from transformers import AutoModelForCausalLM, AutoModel
from models.model_wrapper import ModelWrapper
from dataclasses import dataclass, field
from typing import Optional, Union, Tuple, List
logger = logging.getLogger(__name__)
torch.set_num_threads(1)
from train.arg_classes import ModelArguments, DataArguments, TrainingArguments, make_default_args

# %%
args = tr.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
is_debugger = lambda : sys.gettrace() is not None
use_default_args = is_notebook() or is_debugger() 
model_args, data_args, training_args = args.parse_args_into_dataclasses(make_default_args(use_default_args)) #replace with isnotebook(), if not debugging    model_args.device_map = "cuda:0" if training_args.deepspeed is None else None
model_args.device_map = "cuda:0" if training_args.deepspeed is None else None
checkpoint_dir = ROOT_DIR / "results/model_news_cls_v2"

# %%
assert checkpoint_dir.exists()
checkpoint_dir_epoch = [path for path in checkpoint_dir.iterdir() if path.name.startswith("checkpoint-")][-1]
checkpoint_dir_epoch

# %%


model_args
# %%


model = ModelWrapper(model_args, training_args)
# %%
model
# %%
from huggingface_hub import upload_folder

upload_folder(repo_id = "tanguyrenaudie/Mistral_relevance", folder_path = str(checkpoint_dir_epoch),commit_message="uploading model", token = "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")
# %%
