# DELETE this file


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
import pandas as pd
# %% 


path_data = os.environ["PATH_DATA"]
path_data = Path(path_data)
temp_pseudo_labels_path = next(path_data.glob("temp_pseudo_labels*"))
temp_pseudo_labels_path 
news_headlinews_v3p2_path = next(path_data.glob("news_headlines_v3.2*"))
news_headlinews_v3p2_path
# dfbase = pd.read_parquet(news_headlinews_v3p2_path)
# %% 
df = pd.read_parquet(temp_pseudo_labels_path)   
# %%
# index cluster, DATE, SYMBOL 
import plotly.express as px
fig = px.histogram(df.reset_index("DATE"), x="DATE",color= "cluster", nbins=20)
fig.update_layout(
    title="Distribution of dates in StreetAccount",
    xaxis_title="Date",
    yaxis_title="Count",
    legend_title="Cluster",
)
# %% 
fig.show()
# %%
# print some lines

df.sort_values(by = ["DATE","SYMBOL", "headline"], inplace=True)
pd.set_option('display.max_colwidth', None)
df.iloc[100000:100000+30][["headline_no_ent_v2"]]
# %%
# look at clusters
pd.set_option('display.max_colwidth', None)
dfpseudo = df.loc[df.pseudo_label.notna()] 
dfpseudo.loc[dfpseudo.cluster == 4][["headline_no_ent_v2", "pseudo_label"]].sample(20)
# %%


fig = px.histogram(df,x="RET_5D_pos",color = "cluster")    
# update fig size 
fig.update_layout(
    figsize = (20,10),
    title="Distribution of RET_5D_pos")
fig.show()


# %%
fig = px.histogram(df.loc[df.pseudo_label.isin([0,1,2])],x="pseudo_label",color = "cluster")
fig.show()
# %%
# update neptune 

sys_ids = [1267]
for sys_id in sys_ids:
    run = neptune.init_run(
        with_id = f"LLM-{sys_id}",
    )
    expname = run["setup/exp_name"].fetch()
    run["setup/exp_name"] = f"llama7B gpt_labels"
    run["setup/logits_pooling"] = "avg"
    run.stop()
# %%
