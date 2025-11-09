# Required Modules
import transformers
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
batch_size = config["batch_size"]
n_heads = config["n_heads"]
n_dim = config["n_dim"]
num_layers = config["num_layers"]
max_seq_len = 12
vocab_tokens = config["vocab_tokens"]
_model = None
_tokenizer = None


def _load_model_once():
    """Load model ONCE when first called"""
    global _model, _tokenizer
    
    if _model is None:
        from models import model_loader
        print(f"[Prefill] Loading model on cuda:1...")
        _model = model_loader.get_model("cuda:1")
        _tokenizer = model_loader.get_tokenizer()
        print(f"[Prefill] Model loaded ✓")
    
    return _model, _tokenizer


def prefill_stage(req_id, prompt, page_table):

    torch.cuda.set_device(1)
    model, tokenizer = _load_model_once()
    
    # tokenizing the input
    input = tokenizer( prompt,return_tensors="pt",max_length=120,truncation=True).to(prefill_device)
    max_seq_len = input["input_ids"].shape[1]

    print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] Prefill START {req_id}", flush=True)

    kv_cache = {}
    page_table[req_id] = {
        "kv_cache" : kv_cache
    }

    # Create a custom cache class that intercepts updates
    class MonitoredCache(DynamicCache):
        
        def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
            
            k_tensor = torch.empty([batch_size,n_heads,max_seq_len,n_dim] , dtype=torch.float16, device=decode_device)
            v_tensor = torch.empty_like(k_tensor)
            dst_k_ptr = k_tensor.data_ptr()
            dst_v_ptr = v_tensor.data_ptr()

            kv_cache[layer_idx] = {
                "K_address" : dst_k_ptr,
                "V_address" : dst_v_ptr
            }


            cp.cuda.runtime.memcpyPeer(
                dst_k_ptr, 0,              # destination address (on cuda:0)
                key_states.data_ptr(), 1,              # source address (on cuda:1)
                key_states.numel() * key_states.element_size()  # size in bytes
            )

            if layer_idx == 0:
                mem = cp.cuda.UnownedMemory(dst_k_ptr,  key_states.numel() * key_states.element_size(), owner=None)
                memptr = cp.cuda.MemoryPointer(mem, 0)
            
                cp_arr = cp.ndarray(
                    shape=(1, 8, 12, 128),
                    dtype=cp.float16,
                    memptr=memptr
                )
            
                dst_from_ptr = torch.as_tensor(cp_arr, device=decode_device)
            
                print("Pointer-read slice:", dst_from_ptr.flatten()[:10])
                
                
            cp.cuda.runtime.memcpyPeer(
                dst_v_ptr,   0,            # destination address (on cuda:0)
                value_states.data_ptr(),   1,            # source address (on cuda:1)
                value_states.numel() * value_states.element_size()  # size in bytes
            )

            # print(f" layer {layer_idx} has been copied to the dstinations")
        
            return super().update(key_states, value_states, layer_idx, cache_kwargs)

    past_key_values = MonitoredCache()

    with torch.no_grad():
        outputs = model(
            **input,
            past_key_values=past_key_values,  
            use_cache=True
        )

    logits = torch.empty([batch_size,vocab_tokens],dtype=torch.float16, device=decode_device)
    dst_logits_ptr = logits.data_ptr()

    page_table[req_id] = {
        "last_logit_token" : dst_logits_ptr
    }

    src_logits_ptr = outputs.logits[:,-1,:]

    cp.cuda.runtime.memcpyPeer(
        dst_logits_ptr, 0,
        src_logits_ptr.data_ptr(), 1,
        src_logits_ptr.numel() * src_logits_ptr.element_size()
    )

    torch.cuda.synchronize(prefill_device)
    
    print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] Prefill END {req_id}", flush=True)
    return {"req_id":req_id , "prompt": prompt }



