from transformers import AutoModel, AutoModelForCausalLM, TrainingArguments, AutoConfig
import torch 
from typing import Optional, Union, Tuple, List
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
try: 
    import flash_attn
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False




class ModelWrapper(torch.nn.Module):
    # this model wrapper implements a cls head on top of a CausalLM or simple transformer Model
    def __init__(self, model_args, training_args:TrainingArguments):
        super().__init__()
        self.model_name = model_args.model_name
        self.lora_dim = model_args.lora_dim
        self.lora_dropout = model_args.lora_dropout
        self.use_mlp = model_args.use_mlp
        self.num_labels = model_args.num_labels
        self.use_causal = model_args.use_causal
        self.device_map = model_args.device_map
        self.mlp_dropout = model_args.mlp_dropout
        self.torch_dtype = torch.bfloat16 if training_args.bf16 else torch.float16
        flashattn_supported = model_args.use_flash_attention_2
        ModelClassAuto = AutoModelForCausalLM if self.use_causal else AutoModel 
        self.model = ModelClassAuto.from_pretrained(self.model_name, use_flash_attention_2 =flashattn_supported and HAS_FLASH_ATTN, torch_dtype = self.torch_dtype, trust_remote_code= True, device_map = self.device_map)
        self.final_dim = self.model.config.hidden_size if not self.use_causal else self.model.config.vocab_size
        if self.use_mlp:
            self.cls_head = torch.nn.Sequential(
                torch.nn.Linear(self.final_dim, self.final_dim//2,device=self.device_map),
                torch.nn.Dropout(self.mlp_dropout),
                torch.nn.ReLU(),
                torch.nn.Linear(self.final_dim//2, self.num_labels, device=self.device_map),
            )
        else:
            self.cls_head = torch.nn.Linear(self.final_dim, self.num_labels, device=self.device_map)

        if self.lora_dim > 0 :
            from peft import LoraConfig, get_peft_model
            lora_alpha = 16
            lora_dropout = model_args.lora_dropout
            peft_config = LoraConfig(
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                r=self.lora_dim,
                bias="none",
                target_modules=model_args.target_modules,
            )
            self.model = get_peft_model(self.model, peft_config=peft_config)

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
            batch_size = input_ids.shape[0]
            return_dict = (
                return_dict if return_dict is not None else self.model.config.use_return_dict
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

            if self.model.config.pad_token_id is None:
                sequence_lengths = -1
            else:
                if input_ids is not None:
                    sequence_lengths = (
                        torch.eq(input_ids, self.model.config.pad_token_id).long().argmax(-1) - 1
                    ).to(hidden_states.device)
                else:
                    sequence_lengths = -1

            if hasattr(self.cls_head, "modules_to_save"):
                module_to_save_values = self.cls_head.modules_to_save.values()
                mlp = list(module_to_save_values)[0]
                fc1 = mlp.fc1
                dtype = fc1.weight.dtype
            elif list(self.cls_head.parameters()):
                dtype = list(self.cls_head.parameters())[0].dtype
            else:
                dtype = torch.float32  # default dtype for the final layer
            logits = self.cls_head(hidden_states.to(dtype) )

            # POOLING METHOD
            if question_mask is None:
                question_mask = torch.arange(logits.size(1), device=logits.device).expand(
                    batch_size, logits.size(1)
                ) <= sequence_lengths.unsqueeze(1)
            assert question_mask.shape == logits.shape[:2] #(B, L)
            logits_sum = torch.sum(
                logits * question_mask.unsqueeze(2).to(torch.int), dim=1
            )  # -> B,2
            pooled_logits = logits_sum / question_mask.sum(dim=1, keepdim=True)
            # other methods
            # avg pooling. sequence_lengths is the position of the last VALID token

            loss = None
            if labels is not None:
                labels = labels.to(logits.device)
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(
                    pooled_logits.view(-1, self.num_labels), labels.view(-1)
                )

            if not return_dict:
                output = (pooled_logits,) + transformer_outputs[1:]
                return ((loss,) + output) if loss is not None else output

            return SequenceClassifierOutputWithPast(
                loss=loss,
                logits=pooled_logits,
                past_key_values=transformer_outputs.past_key_values,
                hidden_states=transformer_outputs.hidden_states,
                attentions=transformer_outputs.attentions,
            )
