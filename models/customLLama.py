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



class LlamaConfigCustom(LlamaConfig):
    def __init__(
        self,
        pooler_type_logits: str = "last",
        pooler_type_hidden_states: str = "none",
        debug: bool = False,
        use_mlp: bool = False,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        num_labels: int = 2,
        **kwargs,
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            hidden_act=hidden_act,
            max_position_embeddings=max_position_embeddings,
            initializer_range=initializer_range,
            rms_norm_eps=rms_norm_eps,
            use_cache=use_cache,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pretraining_tp=pretraining_tp,
            tie_word_embeddings=tie_word_embeddings,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            num_labels=num_labels,
            **kwargs,
        )
        self.pooler_type_logits = pooler_type_logits
        self.pooler_type_hidden_states = pooler_type_hidden_states
        self.debug = debug
        self.use_mlp = use_mlp



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

    def load_layer_to_cpu(self, layer_name):
        self.weights_loader.set_state_dict(layer_name, self.device)
        state_dict = self.weights_loader.get_state_dict(self.device)
        if "value_head.weight" in state_dict:
            state_dict = {"lm_head.weight": state_dict["value_head.weight"]}
        return state_dict

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


class LlamaForSequenceClassificationMLP3(nn.Module):
    # this version adds a mask for the pooling type, to only get the average logits for the question tokens
    def __init__(
        self, model_name: str, torch_dtype: torch.dtype | str, config: LlamaConfig
    ):
        super().__init__()
        self.num_labels = config.num_labels
        self.use_mlp = config.use_mlp
        self.pooler_type_logits = config.pooler_type_logits
        self.pooler_type_hidden_states = config.pooler_type_hidden_states
        self.model = LlamaModel.from_pretrained(
            model_name, config=config, torch_dtype=torch_dtype
        )
        self.config = config
        # 4096 for 7Bconfig
        self.score = (
            MLP(config.hidden_size, config.hidden_size // 2, self.num_labels)
            if self.use_mlp
            else nn.Linear(config.hidden_size, self.num_labels)
        ).to(device=self.model.device, dtype=torch_dtype)
        # Initialize weights and apply final processing
        for n, m in self.score.named_modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        question_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (
                    torch.eq(input_ids, self.config.pad_token_id).long().argmax(-1) - 1
                ).to(hidden_states.device)
            else:
                sequence_lengths = -1

        hidden_states = pool_hidden_states(
            hidden_states, sequence_lengths, self.pooler_type_hidden_states
        )
        if hasattr(self.score, "modules_to_save"):
            module_to_save_values = self.score.modules_to_save.values()
            mlp = list(module_to_save_values)[0]
            fc1 = mlp.fc1
            dtype = fc1.weight.dtype
        elif list(self.score.parameters()):
            dtype = list(self.score.parameters())[0].dtype
        else:
            dtype = torch.float32  # default dtype for the final layer
        logits = self.score(hidden_states.to(dtype))

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError(
                "Cannot handle batch sizes > 1 if no padding token is defined."
            )

        if question_mask is None:
            question_mask = torch.arange(logits.size(1), device=logits.device).expand(
                batch_size, logits.size(1)
            ) <= sequence_lengths.unsqueeze(1)
        assert question_mask.shape == logits.shape[:2]

        logits_sum = torch.sum(
            logits * question_mask.unsqueeze(2).to(torch.int), dim=1
        )  # -> B,2
        try:
            assert all(question_mask.sum(dim=1) > 0)
        except AssertionError:
            print(f"division by zero ! {question_mask.sum(dim=1)}")
        pooled_logits = logits_sum / question_mask.sum(dim=1, keepdim=True)
        # other methods
        # avg pooling. sequence_lengths is the position of the last VALID token

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(
                    pooled_logits.view(-1, self.num_labels), labels.view(-1)
                )
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output
        try:
            assert torch.isnan(loss) == False
        except AssertionError:
            print(f"torch isnan : {torch.isnan(loss)}")
            print(f"pooled logits : {pooled_logits}")
            print(f"question mask : {question_mask.sum(dim=1)}")
            raise AssertionError
        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def __repr__(self):
        ret = super().__repr__()
        ret += f"\npooler_types -- hiddenstates: {self.pooler_type_hidden_states},  logits: question_mask_avg"
        return ret


