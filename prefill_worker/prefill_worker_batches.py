import torch
import cupy as cp
from transformers.cache_utils import DynamicCache
from datetime import datetime
import time
import yaml

DEBUG_PREFILL = False

# Loading the Yaml files
with open("config/model_config.yaml", "r") as f:
    config = yaml.safe_load(f)

decode_device = "cuda:0"
prefill_device = "cuda:1"

_model = None
_tokenizer = None

def enable_p2p():
    for i in range(2):
        for j in range(2):
            if i != j:
                with cp.cuda.Device(i):
                    try:
                        cp.cuda.runtime.deviceEnablePeerAccess(j)
                    except:
                        pass

def _load_model_once():
    """Load model ONCE when first called"""
    global _model, _tokenizer
    
    if _model is None:
        from models import model_loader
        print(f"[Prefill] Loading model on cuda:1...") if DEBUG_PREFILL else None
        _model = model_loader.get_model("cuda:1")
        _tokenizer = model_loader.get_tokenizer()
        print(f"[Prefill] Model loaded ✓") if DEBUG_PREFILL else None
    
    return _model, _tokenizer

def prefill_stage(req_id, input_ids , request_prompt,page_table,cpu_kv_manager, kv_write_to_cpu_queue):

    torch.cuda.set_device(1)

    print(f"[Prefill] for req {req_id} has been started ")
    enable_p2p()
    print(f"[Prefill] for req {req_id} has Enable P2P ") if DEBUG_PREFILL else None

    model , tokenizer = _load_model_once()

    seq_len = input_ids.shape[1]
    print(f"Number of tokens in {req_id} is {seq_len} ")

    page_table[req_id]["seq_len"] = seq_len
    page_table[req_id]["req_id_start_time"] = time.time()
    page_table[req_id]["prefill_start_time"] = time.time()

    print(f"Prefill START {req_id}") if DEBUG_PREFILL else None


    class CPUStreamingCache(DynamicCache):
        def update(self, key_states, value_states, layer_idx, cache_kwargs=None):

            k_clone = key_states.detach().contiguous()
            v_clone = value_states.detach().contiguous()
    
            kv_write_to_cpu_queue.put((req_id,layer_idx,k_clone,v_clone))
    
            # print(f"[Prefill] for req {req_id} has sucessfully incremented the layer {page_table.get_layer_at_cpu(req_id)}") if DEBUG_PREFILL and layer_idx == 0 else None 
    
            return super().update(key_states, value_states, layer_idx, cache_kwargs)
    
    cache = CPUStreamingCache()

    print(f"[Prefill] for req {req_id} has started forward pass") if DEBUG_PREFILL else None
    
    # Forward pass
    with torch.no_grad():
        outputs = model(**input_ids, past_key_values=cache, use_cache=True)

    print(f"[Prefill] for req {req_id} has completed forward pass") if DEBUG_PREFILL else None

    torch.cuda.synchronize(prefill_device)

    page_table[req_id]["prefill_end_time"] = time.time()

    

    