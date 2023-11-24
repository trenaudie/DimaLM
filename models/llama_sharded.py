# PIPELINE V12 - for using LlamaCausaltoClassification
# IDEa use the CausalLM then add a clsh

import torch
import torch.nn as nn
import os
import logging
from transformers import LlamaConfig
from transformers import LlamaTokenizer
from transformers import LlamaModel
from transformers import LlamaPreTrainedModel
from transformers import LlamaForCausalLM
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from transformers import AutoModelForCausalLM
from typing import Optional, Tuple, Union, List
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from safetensors.torch import save_model
from accelerate import infer_auto_device_map

from models.dense_layers import MLP
from models.pool_layer import pool_hidden_states, pool_logits

HAS_FLASH_ATTN = True
try:
    from flash_attn.models.gpt import GPTLMHeadModel
    from flash_attn.utils.pretrained import state_dict_from_pretrained
    from flash_attn.models.llama import (
        llama_config_to_gpt2_config,
        config_from_checkpoint,
        remap_state_dict_hf_llama,
    )
except ImportError:
    HAS_FLASH_ATTN = False
    print("flash_attn not installed")

from accelerate.utils import (
    set_module_tensor_to_device,
)
import re
from pathlib import Path
from time import time
from transformers import LlamaTokenizer, TrainingArguments
from accelerate import init_empty_weights
import gc
import ctypes
from safetensors.torch import load_file
from typing import Optional, Tuple, Union, List, Sequence
from utils.io_utils import SuppressLogging


def clean_memory(verbose: bool = False):
    gc.collect()
    ctypes.CDLL("libc.so.6").malloc_trim(0)
    torch.cuda.empty_cache()
    if verbose:
        print(
            f"memory on device 0 = {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB"
        )
        print(
            f"memory on device 1 = {torch.cuda.memory_allocated(1) / 1024 ** 3:.2f} GB"
        )


class TrainingArgsShardedLlama(TrainingArguments):
    @property
    def place_model_on_device(self):
        return False


class ShardedLlama(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        checkpoint_path: Optional[os.PathLike],
        device="cuda:0",
        dtype=torch.bfloat16,
    ):
        """
        Sharded version of LlamaForCausalLM : the model is splitted into layer shards to reduce GPU memory usage.
        During the forward pass, the inputs are processed layer by layer, and the GPU memory is freed after each layer.
        To avoid loading the layers multiple times, we could save all the intermediate activations in RAM, but
        as Kaggle accelerators have more GPU memory than CPU, we simply batch the inputs and keep them on the GPU.

        Parameters
        ----------
        checkpoint_path : str or Path
            path to the checkpoint
        device : str, optional
            device, by default 'cuda:0'
        dtype : torch.dtype, optional
            dtype, by default torch.float16
        """
        super().__init__()
        # Save parameters
        self.checkpoint_path = Path(checkpoint_path)
        self.model_name = getattr(config, "name_or_path", None)
        self.device = device
        self.dtype = dtype

        # Create model
        self.config = config  # classic llama config, not custom
        self.init_model()

        self.layer_names_pretrained = (
            ["model.embed_tokens"]
            + [f"model.layers.{i}" for i in range(len(self.model.model.layers))]
            + ["model.norm"]  # not the lm_head (for classification)
        )
        self.re_patterns_save_module = [
            r"embed_tokens$",
            r"layers.\d+$",
            r"\.norm$",
            r"lm_head$",
        ]

        # check if cekcpoint path has all the layers
        assert self.checkpoint_path.exists(), "checkpoint path does not exist"
        assert self.is_checkpoint_path(
            self.checkpoint_path
        ), "checkpoint path is invalid"

    def init_model(self):
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(self.config)
            self.model = model
            self.model.tie_weights()

        self.lm_head = (
            nn.Linear(self.config.hidden_size, self.config.num_labels)
            .to(self.dtype)
            .to(self.device)
        )  # overrides the lm_head from the model, puts th
        self.layers_pretrained = (
            [self.model.model.embed_tokens]
            + list(self.model.model.layers)
            + [self.model.model.norm]  # not the lm_head !
        )

        # Move buffers to device (not that much GPU memory used)
        for buffer_name, buffer in self.model.named_buffers():
            set_module_tensor_to_device(
                self.model, buffer_name, self.device, value=buffer, dtype=self.dtype
            )

    def is_checkpoint_path(self, checkpoint_path: os.PathLike) -> bool:
        checkpoint_path = Path(checkpoint_path)
        check1 = checkpoint_path.exists()
        check2 = (checkpoint_path / "config.json").exists()
        check3 = (checkpoint_path / "tokenizer_config.json").exists()
        check4 = all(
            filter(
                lambda path: any(path.name in x for x in self.layer_names_pretrained),
                checkpoint_path.iterdir(),
            )
        )
        # check for full layers, not independent operations
        not_full_paths = [
            "k_proj.safetensors",
            "v_proj.safetensors",
            "q_proj.safetensors",
            "mlp.up_proj.safetensors",
        ]
        check5 = not any(
            filter(
                lambda path: any(x in path.name for x in not_full_paths),
                checkpoint_path.iterdir(),
            )
        )
        if any(not check for check in [check1, check2, check3, check4, check5]):
            return False
        # check for actual architecture (80 layers, etc..)
        layers_not_saved = []
        for name, module in self.model.named_modules():
            if any(
                re.search(pattern, name) for pattern in self.re_patterns_save_module
            ):
                # this is a layer i want to save, check that it is saved locally in checkpoin_path directory
                path_layer = checkpoint_path / f"{name}.safetensors"
                path_layer2 = checkpoint_path / f"{name.split('.')[-1]}.safetensors"
                if not path_layer.exists() and not path_layer2.exists():
                    layers_not_saved.append(name)
        return True

    def find_module(self, parent_module: nn.Module, layer_name_re: str):
        for name, module in parent_module.named_modules():
            if name == "":
                continue
            if re.search(layer_name_re, name):
                return module
        return None

    def load_layer(self, layer_name):
        state_dict = load_file(
            self.checkpoint_path / (layer_name + ".safetensors"), device=self.device
        )

        def find_module(model: nn.Module, layer_name_re: str):
            for name, module in model.named_modules():
                if name == "":
                    continue
                if re.search(layer_name_re, name):
                    return module
            return None

        for param_name, param in state_dict.items():
            assert (
                param.dtype != torch.int8
            ), "int8 not supported (need to add fp16_statistics)"
            is_sub_arg_idx = 0
            for i, arg in enumerate(param_name.split(".")):
                if arg in layer_name.split("."):
                    is_sub_arg_idx = i + 1
            tensor_name_args = param_name.split(".")[is_sub_arg_idx:]
            tensor_name = ".".join(tensor_name_args)

            module = find_module(self.model, layer_name)
            set_module_tensor_to_device(
                module, tensor_name, self.device, value=param, dtype=self.dtype
            )

    def saving_model_layers_independently(
        self,
        checkpoint_path: os.PathLike,
        model_config: Optional[LlamaConfig] = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        type_to_string = {
            torch.float32: "fp32",
            torch.float16: "fp16",
            torch.bfloat16: "bf16",
        }
        dtype_str = type_to_string[dtype]
        if not dtype_str in Path(checkpoint_path).name:
            checkpoint_path = Path(checkpoint_path).parent / (
                Path(checkpoint_path).name + f"_{dtype_str}"
            )
            os.makedirs(checkpoint_path, exist_ok=True)
            self.checkpoint_path = checkpoint_path

        device_map = infer_auto_device_map(self.model)

        model = LlamaForCausalLM.from_pretrained(
            self.model_name,
            config=model_config,
            device_map=device_map,
            torch_dtype=dtype,
            offload_folder=checkpoint_path,
            use_safetensors=False,  # very important
        )
        # clear the dir
        for ckpt_file in checkpoint_path.iterdir():
            ckpt_file.unlink()
        os.makedirs(checkpoint_path, exist_ok=True)

        for name, param in model.named_parameters():
            print(name, param.device)
        for key, val in device_map.items():
            print(key, val)
        for layer_name, layer in model.named_modules():
            if any(
                re.search(pattern, layer_name)
                for pattern in self.re_patterns_save_module
            ):
                path_layer = checkpoint_path / f"{layer_name}.safetensors"
                if not path_layer.exists():
                    print(layer_name)
                    save_model(layer, path_layer)

        model.config.save_pretrained(checkpoint_path)
        tokenizer = LlamaTokenizer.from_pretrained(self.model_name)
        tokenizer.save_pretrained(checkpoint_path)

    def __call__(
        self,
        input_ids,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        logger = logging.getLogger("root")

        # Reboot the model to make sure buffers are loaded and memory is clean
        inputs = input_ids
        del self.model
        clean_memory()
        with SuppressLogging():
            self.init_model()

        # Send batch to device
        inputs = inputs.to(self.device)
        MAX_LENGTH = inputs.shape[1]
        BATCH_SIZE = inputs.shape[0]

        # Create attention mask for the largest input, and position ids to use KV cache
        attention_mask = torch.finfo(self.dtype).min * torch.ones(
            MAX_LENGTH, MAX_LENGTH
        )
        attention_mask = attention_mask.triu(diagonal=1)[None, None, ...]
        attention_mask = attention_mask.to(self.device)
        attention_mask = attention_mask.expand(BATCH_SIZE, 1, MAX_LENGTH, MAX_LENGTH)
        position_ids = torch.arange(MAX_LENGTH, dtype=torch.long, device=self.device)[
            None, :
        ]

        sequence_lengths = (
            torch.eq(inputs, self.config.pad_token_id).long().argmax(-1) - 1
        ).to(self.device)
        # with ThreadPoolExecutor() as executor, torch.inference_mode():

        #     # Load first layer
        #     future = executor.submit(self.load_layer, 'model.embed_tokens')
        assert next(self.lm_head.parameters()).requires_grad == True
        with torch.no_grad():
            for i, (layer_name, layer) in enumerate(
                zip(self.layer_names_pretrained, self.layers_pretrained)
            ):
                self.load_layer(layer_name)
                logger.info(
                    f"layer_name: {layer_name}, inputs.shape: {inputs.shape}, gpu {torch.cuda.memory_allocated(self.device) / 1e9:.1f}GB"
                )

                if "embed_tokens" in layer_name:
                    # to receive the output of the embedding (change of shape) - create new tensor on the gpu
                    batch_on_gpu = torch.zeros(
                        BATCH_SIZE,
                        MAX_LENGTH,
                        layer.weight.size(1),
                        device=self.device,
                        dtype=layer.weight.dtype,
                    )
                    for i, sample in enumerate(inputs):
                        batch_on_gpu[i] = layer(sample)
                    inputs = batch_on_gpu
                elif layer_name == "model.norm":
                    # only keep last token (last valid token)
                    # because this is the last layer.
                    last_token_inputs = inputs[
                        torch.arange(BATCH_SIZE), sequence_lengths
                    ]
                    for i, sample in enumerate(inputs):
                        inputs[i] = layer(
                            last_token_inputs[i]
                        )  # [1,4096] applied to [26,4096] (broadcasting)
                else:
                    for i, sample in enumerate(inputs):
                        temp = layer(
                            sample.unsqueeze(0),
                            attention_mask=attention_mask[i : i + 1],
                            position_ids=position_ids,
                        )[
                            0
                        ]  # [1,4K]
                        inputs[i] = temp.squeeze(0)

                for module_name, module in layer.named_modules():
                    if module._parameters:
                        for param_name, param in module.named_parameters():
                            set_module_tensor_to_device(
                                module,
                                param_name,
                                "meta",
                                value=param,
                                dtype=self.dtype,
                            )
                layer.to("meta")
                del layer
                clean_memory()  # Added my CPMP to release memory after each layer is processed
                torch.cuda.empty_cache()
        out = self.lm_head(inputs)
        out = pool_logits(out, sequence_lengths, "avg", self.config.num_labels)
        assert out.shape == (BATCH_SIZE, self.config.num_labels)
        assert labels.shape == (BATCH_SIZE,) or labels.shape == (
            BATCH_SIZE,
            1,
        )
        loss = nn.CrossEntropyLoss()(out, labels.to(out.device))
        logger.warning(f"loss: {loss}")
        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=out,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )


