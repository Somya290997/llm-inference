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


def kv_cache_allocator(req_id, max_seq_len, page_table):
    
    torch.cuda.set_device(0)
    kv_cache = {} 
    
    for layer_idx in range(int(num_layers)):
        # allocate empty space
        k_tensor = torch.empty([batch_size,n_heads,max_seq_len,n_dim] , dtype=torch.float16, device=decode_device)
        v_tensor = torch.empty_like(k_tensor)

        # address ptr
        k_ptr = k_tensor.data_ptr()
        
        if layer_idx == 0:
            print(f"The kv_cache_allocation {k_ptr}")
            
        v_ptr = v_tensor.data_ptr()

        kv_cache[layer_idx] = {
            "K_address" : k_ptr,
            "V_address" : v_ptr,
            "dtype" : torch.float16,
            "max_seq_len" : max_seq_len
        }

        # print(f"KV cache of layer {layer_idx} is placed on {decode_device}")
    final_layer_logits = torch.empty([batch_size,vocab_tokens],dtype=torch.float16, device=decode_device)
    

    page_table[req_id] = {
        "kv_cache" : kv_cache,
        "last_logit_layer" : final_layer_logits.data_ptr()
    }

    if req_id in page_table:
        print(f"are there any keys in this in KV cache allocator {page_table[req_id].keys()}")
    else:
        print(f"[Error] req_id {req_id} not found in page_table_entry")




def prefill_stage(req_id, prompt, page_table):

    torch.cuda.set_device(1)
    model, tokenizer = _load_model_once()
    
    # tokenizing the input
    input = tokenizer( prompt,return_tensors="pt",max_length=120,truncation=True).to(prefill_device)

    # allocate KV cache
    max_seq_len = input["input_ids"].shape[1]

    kv_cache_allocator(req_id,max_seq_len,page_table)

    print(f"KV cache is allocated in {decode_device}")
    print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] Prefill START {req_id}", flush=True) 
    
    if req_id in page_table:
        print(f"are there any keys in this in prefill  {page_table[req_id].keys()}")
    else:
        print(f"[Error] req_id {req_id} not found in page_table_entry")

    # Create a custom cache class that intercepts updates
    class MonitoredCache(DynamicCache):
        
        def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
            
            if layer_idx == 0:
                print(key_states.shape)
                print(key_states.flatten()[:10])
                elem_bytes = key_states.element_size()   # should be 2 for fp16
                numel = key_states.numel()
                bytes_to_copy = elem_bytes * numel
                print(bytes_to_copy) 
                            
            # Distination of the KV caches
            try:
                dst_k_ptr = page_table[req_id]["kv_cache"][layer_idx]["K_address"]
                if layer_idx == 0:
                    print(dst_k_ptr)
                    
                dst_v_ptr = page_table[req_id]["kv_cache"][layer_idx]["V_address"]
                
            except KeyError as e:
                print(f"[ERROR] Missing KV cache entry: {e}")
                return super().update(key_states, value_states, layer_idx, cache_kwargs)
                

            cp.cuda.runtime.memcpyPeer(
                dst_k_ptr, 0,              # destination address (on cuda:0)
                key_states.data_ptr(), 1,              # source address (on cuda:1)
                key_states.numel() * key_states.element_size()  # size in bytes
                                 # kind = 3 → device-to-device
            )

            # time.sleep(10)
            torch.cuda.synchronize(1)
            torch.cuda.synchronize(0)

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
                     # kind = 3 → device-to-device
            )

            torch.cuda.synchronize(prefill_device)

            # print(f" layer {layer_idx} has been copied to the dstinations")
        
            return super().update(key_states, value_states, layer_idx, cache_kwargs)

    past_key_values = MonitoredCache()

    with torch.no_grad():
        outputs = model(
            **input,
            past_key_values=past_key_values,  
            use_cache=True
        )

    dst_logits_ptr = page_table[req_id]["last_logit_layer"]
    src_logits_ptr = outputs.logits[:,-1,:]
    print(f" Shape {src_logits_ptr.shape}")

    cp.cuda.runtime.memcpyPeer(
        dst_logits_ptr, 0,
        src_logits_ptr.data_ptr(), 1,
        src_logits_ptr.numel() * src_logits_ptr.element_size()
    )

    torch.cuda.synchronize(prefill_device)
    
    print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] Prefill END {req_id}", flush=True)
    return {"req_id":req_id , "prompt": prompt }



