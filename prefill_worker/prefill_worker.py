# Required Modules
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from transformers.cache_utils import DynamicCache
from datetime import datetime
import time
from models import model_loader
import yaml
import cupy as cp

# Loading the Yaml files
with open("config/model_config.yaml", "r") as f:
    config = yaml.safe_load(f)


# Loading yaml variable and model caching
decode_device = "cuda:0"
prefill_device = "cuda:1"


# loading model only once 
model = model_loader.get_model(prefill_device)
tokenizer = model_loader.get_tokenizer(prefill_device)

def prefill_stage(req_id,prompt,page_table):

    # tokenizing the input
    input = tokenizer(prompt, return_tensors="pt")

   
    print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] Prefill START {req_id}", flush=True)

    # Create a custom cache class that intercepts updates
    class MonitoredCache(DynamicCache):
        
        def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
            

            # Distination of the KV caches
            dst_k_ptr = page_table[req_id]["kv_cache"][layer_idx]["K_address"]
            dst_v_ptr = page_table[req_id]["kv_cache"][layer_idx]["V_address"]

            cp.cuda.runtime.memcpy(
                dst_k_ptr,               # destination address (on cuda:0)
                key_states.data_ptr(),               # source address (on cuda:1)
                key_states.numel() * key_states.element_size(),  # size in bytes
                3                      # kind = 3 → device-to-device
            )

            cp.cuda.runtime.memcpy(
                dst_v_ptr,               # destination address (on cuda:0)
                value_states.data_ptr(),               # source address (on cuda:1)
                value_states.numel() * value_states.element_size(),  # size in bytes
                3                      # kind = 3 → device-to-device
            )
        
            return super().update(key_states, value_states, layer_idx, cache_kwargs)

    past_key_values = MonitoredCache()

    with torch.no_grad():
        outputs = model(
            **input,
            past_key_values=past_key_values,  
            use_cache=True
        )

    dst_logits_ptr = page_table[req_id]["last_logit_layer"]

    cp.cuda.runtime.memcpy(
        dst_logits_ptr,               
        outputs.logits[:,-1,:].lo.data_ptr(),               
        outputs.logits[:,-1,:].numel() * outputs.logits[:,-1,:].element_size(),    3                     
    )

    print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] Prefill END {req_id}", flush=True)
    return req_id


