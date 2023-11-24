# LLM project RAM 

This project aims to test the performance of LLMs on the news for stock market prediction. 

An implementation of binary classification is proposed. 

Details : 
- Date Submitted 24.11.2023
- Data : Street Account or Dow Jones
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


### Data 
- data_preprocess/data_v3/data_utils_v3.2.ipynb : Street Account preprocessing script
- data_preprocess/data_v6/clusters.py : Street Account clustering script, after preprocessing
- data_preprocess/data_v6/pseudo_labels_gpt.py : Street Account GPT-Labelling script
 

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


