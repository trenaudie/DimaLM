import neptune
import transformers
import transformers as tr
import os
import psutil
import neptune
import torch.nn as nn 
import torch
from typing import Optional
import neptune
import logging
import configparser
import sys
from dotenv import load_dotenv
ROOT_DIR = os.path.abspath("../")
if not ROOT_DIR.endswith("LLM_project2"):
    ROOT_DIR = os.path.abspath("./")
sys.path.append(ROOT_DIR)
load_dotenv(os.path.join(ROOT_DIR, ".env"))


def update_neptune_exp_name(run_id: int, exp_name: str):
    run = neptune.init_run(
        project=os.environ["NEPTUNE_PROJECT"],  
        api_token=os.environ["NEPTUNE_API_TOKEN"],
        with_id=f"LLM-{run_id}",
    )
    # exp_name = run["setup/exp_name"].fetch()
    run["setup/exp_name"] = exp_name
    run.stop()





def log_run(
    run: neptune.Run,
    model: nn.Module,
    training_args: transformers.TrainingArguments,
    model_args,
    data_args,
    len_dataloader_train: int,
    HAS_FLASH_ATTN: bool,
    dataset_train: Optional[torch.utils.data.Dataset] = None,
    filename_headlines: Optional[str] = None,
):
    """
    Logs details to a Neptune run.
    """

    print(f"inside log run for sys id {run['sys/id'].fetch()}")
    model_name = model_args.model_name
    run["setup/model_architecture"] = model_name

    # Create the experiment name
    exp_name_final = training_args.exp_name + " DEBUG" if training_args.is_debug else training_args.exp_name
    run["setup/exp_name"] = exp_name_final

    # Upload model architecture to Neptune
    run["setup/archi"].upload(neptune.types.File.from_content(repr(model)))

    # Log experiment arguments
    training_args_dict = vars(training_args)
    training_args_dict = {k: str(v) for k, v in training_args_dict.items()}
    model_args_dict = vars(model_args)
    model_args_dict = {k: str(v) for k, v in model_args_dict.items()}
    data_args_dict = vars(data_args)
    data_args_dict = {k: str(v) for k, v in data_args_dict.items()}
    run["setup/training_args"] = training_args_dict
    run["setup/model_args"] = model_args_dict
    run["setup/data_args"] = data_args_dict

    if dataset_train is not None:
        if hasattr(dataset_train, "prompt_examples"):
            run["setup/prompt_examples"] = {
                str(k): str(prompt)
                for k, prompt in enumerate(dataset_train.prompt_examples)
            }

    # Log command line details
    us = psutil.Process(os.getpid())
    CMDLINE = us.cmdline()
    print(f"CMDLINE {CMDLINE}")
    assert len(CMDLINE) > 0, "CMDLINE empty"
    run["setup/command_line"] = str(CMDLINE)

    # Compute and log other details
    num_cuda_devices = torch.cuda.device_count()
    cuda_device_ids = list(str(i) for i in range(num_cuda_devices))
    gpu_count = len(cuda_device_ids)

    run["setup/steps_per_epoch"] = len_dataloader_train / (
        training_args.gradient_accumulation_steps * gpu_count
    )
    run["setup/filename_headlines"] = filename_headlines

    # Fetch and return the system ID
    sys_id = run["sys/id"].fetch()
    return sys_id

