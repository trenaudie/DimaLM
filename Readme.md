# LLM project for CEMEF Research 
Project done under 2nd year Mines Paris PSL studies with Prof. David Ryckelynck.

This project aims to test the performance of LLMs, in speed and memory. 
A number of optimizations are benchmarked and profiled: 
- Gradient checkpointing, accumulation and batch size
- Data-parallel training 
- Model-parallel training 
- DeepSpeed ZeRO.
- Sharding and inference of a 70B model to run it layer-by-layer over consumer GPUs

An implementation of binary classification is proposed. 

Details : 
- Date Submitted 24.06.2023 
- Models : 
    - Llama7B, Mistral7B max for training. 
    - LLama70B for sharded training

### Installation 

The following bash script creates a new python environment, activates it and installs the dependencies.

```bash
python3 -m venv venv 
source venv/bin/activate
./install.sh
```
### Configuration 
For each model, you must create a config file located in models/model_confs.json. 
- lora : whether the model supports lora training
- use_flash_attention_2 : whether the model supports Flash Attention 2. Any Llama2 or Mistral variant will support Llama2 


### Training 
The training args inherit from transformers.TrainingArguments. Any argument used in TrainingArguments can be used here. 
I have added some extra arguments: 
Data Arguments
- x_col : str 
- y_col : str
- filename_headlines : str - the name of file in the data folder ex. news_headlines_v3.2.parquet
- add_context : bool - whether to add the past headlines and corresponding returns as context to the prompt. Default is True
Model Arguments
- model_name : str - full name as seen on Huggingface. Ex. "meta-llama/Llama-2-7b-hf
- lora_dim : int - rank of the lora modules 
- use_causal : bool - whether to use a AutoModelForCausalLM or AutoModel as the base model class.


