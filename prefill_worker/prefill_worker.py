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


# # loading model only once 
# model = model_loader.get_model(prefill_device)
# tokenizer = model_loader.get_tokenizer()

def prefill_stage(req_id, prompt, page_table_entry, model, tokenizer):
    
    # tokenizing the input
    input = tokenizer(prompt, return_tensors="pt", max_length=119, truncation=True)
    # MOVE INPUT TO PREFILL DEVICE (NEW)
    input = {k: v.to(prefill_device) for k, v in input.items()}

   
    print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] Prefill START {req_id}", flush=True)

    # Create a custom cache class that intercepts updates
    class MonitoredCache(DynamicCache):
        
        def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
            
            # print(f"Page table for {req_id} at layer {layer_idx}: ", page_table.get(req_id))
            
            # Distination of the KV caches
            try:
                dst_k_ptr = page_table_entry["kv_cache"][layer_idx]["K_address"]
                dst_v_ptr = page_table_entry["kv_cache"][layer_idx]["V_address"]
            except KeyError as e:
                print(f"[ERROR] Missing KV cache entry: {e}")
                return super().update(key_states, value_states, layer_idx, cache_kwargs)

            with cp.cuda.Device(1):  # Source device: prefill GPU
                cp.cuda.runtime.memcpy(
                    dst_k_ptr,
                    key_states.data_ptr(),
                    key_states.numel() * key_states.element_size(),
                    3
                )
        
                cp.cuda.runtime.memcpy(
                    dst_v_ptr,
                    value_states.data_ptr(),
                    value_states.numel() * value_states.element_size(),
                    3
                )

            # cp.cuda.runtime.memcpy(
            #     dst_k_ptr,               # destination address (on cuda:0)
            #     key_states.data_ptr(),               # source address (on cuda:1)
            #     key_states.numel() * key_states.element_size(),  # size in bytes
            #     3                      # kind = 3 → device-to-device
            # )

            # cp.cuda.runtime.memcpy(
            #     dst_v_ptr,               # destination address (on cuda:0)
            #     value_states.data_ptr(),               # source address (on cuda:1)
            #     value_states.numel() * value_states.element_size(),  # size in bytes
            #     3                      # kind = 3 → device-to-device
            # )
        
            return super().update(key_states, value_states, layer_idx, cache_kwargs)

    past_key_values = MonitoredCache()

    with torch.no_grad():
        outputs = model(
            **input,
            past_key_values=past_key_values,  
            use_cache=True
        )

    # dst_logits_ptr = page_table[req_id]["last_logit_layer"]

    # with cp.cuda.Device(1):
    #     cp.cuda.runtime.memcpy(
    #         dst_logits_ptr,
    #         outputs.logits[:,-1,:].data_ptr(),
    #         outputs.logits[:,-1,:].numel() * outputs.logits[:,-1,:].element_size(),
    #         3
    #     )
    logits_slice = outputs.logits[:, -1, :].contiguous()
    dst_logits_ptr = page_table_entry["last_logit_layer"]
    
    # Copy logits from cuda:1 to cuda:0 via P2P
    with cp.cuda.Device(1):
        cp.cuda.runtime.memcpy(
            dst_logits_ptr,
            logits_slice.data_ptr(),
            logits_slice.numel() * logits_slice.element_size(),
            3
        )
        # SYNCHRONIZE AFTER P2P COPY (NEW)
        torch.cuda.current_stream(prefill_device).synchronize()
    
    print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] Prefill END {req_id}", flush=True)
    return req_id



