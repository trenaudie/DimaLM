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
from models.load_model_v14 import load_model_pretrained
from utils.pipeline_utils import is_notebook, set_handlers
from data_modules import DataCollatorCustom, load_dataset_pseudo_label
from arg_classes import ModelArguments, DataArguments, TrainingArguments, make_default_args
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from transformers import AutoModelForCausalLM, AutoModel
from models.model_wrapper import ModelWrapper
try:
    import flash_attn
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False
from dataclasses import dataclass, field
from typing import Optional, Union, Tuple, List
logger = logging.getLogger(__name__)


def compute_metrics(pred):
    # accuracy
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = (labels == preds).mean()
    return {"accuracy": acc}


# %%
print(f"torch version cuda {torch.version.cuda}")


# %% 
def is_debugger():
    return sys.gettrace() is not None

def train():
    pass
    # %%
    args = tr.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    use_default_args = is_notebook() or is_debugger() 
    model_args, data_args, training_args = args.parse_args_into_dataclasses(make_default_args(use_default_args)) #replace with isnotebook(), if not debugging    model_args.device_map = "cuda:0" if training_args.deepspeed is None else None
    model_args.device_map = "cuda:0" if training_args.deepspeed is None else None


   


    # %% 
    model_args.use_causal = False 
    model = ModelWrapper(model_args, training_args)

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



    # %% 
    collator_fn = DataCollatorCustom(tokenizer=dataset_full["train"].tokenizer)
    assert training_args.evaluation_strategy == "steps"
    assert training_args.eval_steps >= 1 and training_args.eval_steps <= 100
    assert len(dataset_full["validation"])>1e3 or (training_args.is_debug and len(dataset_full["validation"])>10)
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
        training_args,
        model_args,
        data_args,
        len_dataloader_train,
        HAS_FLASH_ATTN,
        dataset_full["train"],
        filename_headlines=data_args.filename_headlines,
    )

    os.makedirs(training_args.output_dir, exist_ok=True)
    logger = logging.getLogger("root")
    logger = set_handlers(logger, sys_id, ROOT_DIR)  # creates a file handler.


    # %%

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
    batch = next(itertools.islice(iter(dataloader2), 10, 11))
    print(f"batch keys {batch.keys()}")

    dataloader3 = trainer.get_eval_dataloader()
    batch = next(itertools.islice(iter(dataloader3), 10, 11))
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

if __name__ == "__main__":
    train()