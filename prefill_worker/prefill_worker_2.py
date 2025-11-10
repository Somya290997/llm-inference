import torch
import cupy as cp
from transformers.cache_utils import DynamicCache
from datetime import datetime
from cpu_kv_manger.cpu_kv_blockmanger import CPUKVBlockManager
from common.page_table_2 import PageTable
import yaml

_cpu_kv_manager = CPUKVBlockManager()

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
        print(f"[Prefill] Loading model on cuda:1...")
        _model = model_loader.get_model("cuda:1")
        _tokenizer = model_loader.get_tokenizer()
        print(f"[Prefill] Model loaded ✓")
    
    return _model, _tokenizer

def prefill_stage(req_id,prompt,page_table):

    enable_p2p()
    model , tokenizer = _load_model_once()

    inputs = tokenizer(prompt, return_tensors="pt").to(prefill_device)
    seq_len = inputs["input_ids"].shape[1]

    page_table.init_request(req_id = req_id, num_layers=model.config.num_hidden_layers)

    hidden_dim = model.config.hidden_size // model.config.num_attention_heads
    kv_shape = (1, model.config.num_attention_heads, seq_len, hidden_dim)

    _cpu_kv_manager.alloc_for_request( req_id=req_id, 
                                      num_layers = model.config.num_hidden_layers, 
                                      shape = kv_shape, 
                                      dtype = torch.float16)
    
    print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] Prefill START {req_id}")

    class CPUStreamingCache(DynamicCache):
        def update(self, key_states, value_states, layer_idx, cache_kwargs=None):

            # Clone to keep alive beyond layer update
            k_clone = key_states.contiguous().detach().to("cpu")
            v_clone = value_states.contiguous().detach().to("cpu")

            _cpu_kv_manager.write_layer(req_id, layer_idx, k_clone, v_clone)
            page_table.set_cpu_kv(req_id, layer_idx, k_clone, v_clone)

            if layer_idx == 0:
                print(f"[Prefill] Layer 0 CPU KV written. slice={k_clone.flatten()[:10]}")

            return super().update(key_states, value_states, layer_idx, cache_kwargs)

    cache = CPUStreamingCache()

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs, past_key_values=cache, use_cache=True)

    logits_cpu = outputs.logits[:, -1, :].detach().to("cpu")
    page_table.table[req_id]["logits_cpu"] = logits_cpu

    print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] Prefill END {req_id}")
    return logits_cpu


    










