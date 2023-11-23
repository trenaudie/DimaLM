# %% 
from dataclasses import dataclass, field
from typing import Optional
import transformers as tr
import json, itertools
from pathlib import Path
ROOT_DIR = next(filter(
    lambda p: "LLM_project" in p.name, Path.cwd().iterdir().__next__().parents), None)  
assert ROOT_DIR.exists()


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
    use_causal : bool = field(default=False)
    def __post_init__(self):
        model_specific_args_fixed = dict(json.load(open(ROOT_DIR/"models/model_confs.json", "r")))[self.model_name]
        for k,v in model_specific_args_fixed.items():
            print(f"inside ModelArguments: setting {k} to {v}")
            setattr(self, k, v)
        if hasattr(self,"use_lora"):
            if not self.use_lora: 
                assert self.lora_dim == 0






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
    add_context : bool = field(
        default=False,
        metadata={"help": "Whether to add context to the input prompt (ie past headlines and target)."},
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
    ddp_find_unused_parameters: bool = field(default=False)





def make_default_args(use_default_args:bool):
    # fetch the default values in default_args.json
    if use_default_args:
        default_args_path = ROOT_DIR/"tests/test_args_v2.json"
        assert default_args_path.exists(), f"Default args file {default_args_path} does not exist"
        with open(default_args_path, "r") as fp:
            default_args_dict = json.load(fp)
        default_args = [(f"--{k}", str(v)) for k, v in default_args_dict.items()]
        default_args = list(itertools.chain(*default_args))
        return default_args 
    else:
        return None