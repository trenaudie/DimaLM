### this pipeline implements a Sharded Llama using a checkpoint_path dir that has each layer saved to disk.
import os
import sys
import warnings
import torch
import tokenizers
import transformers


print(f"tokenizers.__version__: {tokenizers.__version__}")
print(f"transformers.__version__: {transformers.__version__}")

from transformers import AutoConfig

import transformers as tr
from transformers import (
    AutoConfig,
)
import argparse
import bitsandbytes as bnb
import numpy as np
from itertools import chain
from pathlib import Path

ROOT_DIR = next(
    filter(
        lambda s: "LLM" in s.name, chain(Path().absolute().parents, [Path(os.getcwd())])
    ),
    None,
)
print(f"ROOT_DIR {ROOT_DIR}")
sys.path.append(str(ROOT_DIR))
from models.customMistral import MistralConfigCustom, MistralForSeqClassCustom
from transformers import BertConfig, BertForSequenceClassification
from transformers import BertForSequenceClassification, BertConfig
from transformers import TrainingArguments
from models.customLLama import (
    LlamaConfigCustom,
    LlamaForSequenceClassificationMLP3,
    ShardedLlama,
)
from peft import get_peft_model, LoraConfig

try:
    import flash_attn
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False


def load_model_pretrained(
    model_args: argparse.Namespace,
    training_args: TrainingArguments,
    mlp_version: int = 3,
):
    """
    Loading a model based on a model_name.
    Parameters
    - model_args : config for the model, must contain model_name attribute
        - model_name: one of the following
            - meta-llama/Llama-2-7b-hf
            - meta-llama/Llama-2-70b-hf
            - PY007/TinyLlama-1.1B-step-50K-105b
            - mistralai/Mistral-7B-v0.1
            - bert-base-uncased

    - training_args : config for training, must contain is_debug attribute
    """
    assert model_args.model_name is not None, "model_name is None"
    model_name = model_args.model_name
    model = None
    print(f"model_name: {model_name}, device_map {model_args.device_map}")
    if "llama-2-7b" in model_name.lower() or "tinyllama" in model_name.lower():
        config_llama = LlamaConfigCustom.from_pretrained(
            model_name,
            pooler_type_logits=model_args.pooler_type_logits,
            debug=training_args.is_debug,
            use_mlp=model_args.use_mlp,
            pad_token_id=2,
            device_map=model_args.device_map,
            torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float16,
            num_labels=model_args.num_labels,
        )
        config_llama.use_cache = False  # maybe i can add KV cache later
        config_llama._flash_attn_2_enabled = HAS_FLASH_ATTN
        print(f"config_llama num labels {config_llama.num_labels}")
        LlamaClass = None
        if mlp_version == 0:
            # LlamaClass = LlamaForSequenceClassificationMLP
            raise NotImplementedError("MLP0 not supported")
        elif mlp_version == 3:
            LlamaClass = LlamaForSequenceClassificationMLP3
        else:
            raise NotImplementedError(f"mlp_version {mlp_version} not supported")
        model = LlamaClass(
            model_name,
            config=config_llama,
            torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float16,
        )
    elif "llama-2-70b" in model_name.lower():
        config = AutoConfig.from_pretrained(
            "garage-bAInd/Platypus2-70B",  # hard coded for now
            pad_token_id=2,
        )
        config.use_cache = False  # maybe i can add KV cache later
        checkpoint_path = ROOT_DIR / "model_weights/platypus2"
        assert checkpoint_path.exists()
        model = ShardedLlama(
            config=config,
            checkpoint_path=checkpoint_path,
            dtype=torch.bfloat16,
        )
    elif "mistral" in model_name.lower():
        mistral_config = MistralConfigCustom.from_pretrained(
            model_name,
            pooler_type_logits=model_args.pooler_type_logits,
            debug=training_args.is_debug,
            use_mlp=model_args.use_mlp,
            pad_token_id=2,
            device_map=model_args.device_map,
        )
        model = MistralForSeqClassCustom.from_pretrained(
            model_name,
            config=mistral_config,
            use_flash_attention_2=HAS_FLASH_ATTN,
            torch_dtype = torch.bfloat16 if training_args.bf16 else torch.float16,
        )


    elif "bert" in model_name.lower():
        config = BertConfig.from_pretrained(
            pretrained_model_name_or_path="bert-base-uncased",
            pad_token_id=0,
        )
        model = BertForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path="bert-base-uncased",
            config=config,
            device_map=model_args.device_map,
        )
    elif "yi" in model_name.lower():
        print(f"model_name {model_name} - raw from huggingface (no pooler)")
        from transformers import AutoModelForSequenceClassification, AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float16,
            device_map=model_args.device_map,
            use_flash_attention_2=HAS_FLASH_ATTN,
            trust_remote_code=True)
    else:
        print(f"model_name {model_name} - raw from huggingface (no pooler)")
        from transformers import AutoModelForSequenceClassification
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float16,
            device_map=model_args.device_map,
            use_flash_attention_2=HAS_FLASH_ATTN,
            trust_remote_code=True,
        )
    # assert model.config.use_cache == False checking if this is ok
    if model_args.lora_dim > 0 and "70b" not in model_args.model_name.lower():
        # do not add lora to Llama 70B
        lora_alpha = 16
        lora_dropout = model_args.lora_dropout
        peft_config = LoraConfig(
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            r=model_args.lora_dim,
            bias="none",
            task_type="SEQ_CLS",
            target_modules=model_args.target_modules,
        )
        model = get_peft_model(model, peft_config=peft_config)
    return model
