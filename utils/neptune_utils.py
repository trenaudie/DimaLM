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
    model_name: str,
    args,
    len_dataloader_train: int,
    HAS_FLASH_ATTN: bool,
    dataset_train: Optional[torch.utils.data.Dataset] = None,
    filename_headlines: Optional[str] = None,
):
    """
    Logs details to a Neptune run.

    Args:
        run: Neptune run object
        model: The model to log
        model_name: Name of the model architecture
        args: Argument object
        HAS_FLASH_ATTN: A boolean indicating if flash attention is available

    Returns:
        sys_id: ID of the system run in Neptune
    """


    run["setup/model_architecture"] = model_name

    # Create the experiment name
    exp_name_final = args.exp_name + " DEBUG" if args.is_debug else args.exp_name
    run["setup/exp_name"] = exp_name_final

    # Upload model architecture to Neptune
    run["setup/archi"].upload(neptune.types.File.from_content(repr(model)))

    # Log experiment arguments
    run["setup/args"] = vars(args)

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
    run["setup/command_line"] = CMDLINE

    # Compute and log other details
    num_cuda_devices = torch.cuda.device_count()
    cuda_device_ids = list(str(i) for i in range(num_cuda_devices))
    gpu_count = len(cuda_device_ids)

    run["setup/steps_per_epoch"] = len_dataloader_train / (
        args.gradient_accumulation_steps * gpu_count
    )
    run["setup/filename_headlines"] = filename_headlines

    # Fetch and return the system ID
    sys_id = run["sys/id"].fetch()
    return sys_id

