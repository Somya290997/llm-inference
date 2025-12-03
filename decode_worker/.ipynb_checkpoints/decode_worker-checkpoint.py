# Required Modules
import os
import torch
from transformers.cache_utils import DynamicCache
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime
import time
import yaml
from utils import wrap_ptr_to_tensor
from models import model_loader

# load the yaml file
with open("config/model_config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Now you can access values:
batch_size = config["batch_size"]
n_heads = config["n_heads"]
n_dim = config["n_dim"]
num_layers = config["num_layers"]
max_seq_len = 12
max_output_len = 100
decode_device = "cuda:0"
prefill_device = "cuda:1"
vocab_tokens = config["vocab_tokens"]

_model = None
_tokenizer = None

def _load_model_once():
    """Load model ONCE when first called"""
    global _model, _tokenizer
    
    if _model is None:
        from models import model_loader
        print(f"[Decode] Loading model on cuda:0...")
        _model = model_loader.get_model("cuda:0")
        _tokenizer = model_loader.get_tokenizer()
        print(f"[Decode] Model loaded ✓")
    
    return _model, _tokenizer

# KV cache creator
def kv_cache_collection(req_id,num_layers,page_table):
    temp_kv = page_table[req_id]["kv_cache"]
    past_key_values = DynamicCache()

    for i in range(0,num_layers):

        src_k_address = temp_kv[i]["K_address"]
        if i == 0:
            print(src_k_address)
        src_v_address = temp_kv[i]["V_address"]
        shape = (batch_size,n_heads,max_seq_len,n_dim)
        dtype = torch.float16

        key_states = wrap_ptr_to_tensor.wrap_ptr_to_tensor(src_k_address,shape,dtype,decode_device)
        value_states = wrap_ptr_to_tensor.wrap_ptr_to_tensor(src_v_address,shape,dtype,decode_device)

        past_key_values.update(key_states,value_states,i,cache_kwargs=None)
        
    return past_key_values

#  Last token logits layers
def last_token_logits_collection(req_id,page_table):

    temp_logit_ptr = page_table[req_id]["last_logit_layer"]
    shape = (batch_size,vocab_tokens)
    dtype = torch.float16
    logit_layer = wrap_ptr_to_tensor.wrap_ptr_to_tensor(temp_logit_ptr,shape,dtype,decode_device)
    return logit_layer

def decode_phase(req_id , page_table):
    
    torch.cuda.set_device(0)
    model, tokenizer = _load_model_once()
    print("KV formation started in Decode")
    formed_KV_caches = kv_cache_collection(req_id,num_layers,page_table)
    print("KV Completed started in Decode")
    K, V = formed_KV_caches[0]
    print(K.flatten()[:10])
    print(f"KV shape {K.shape} and {V.shape}")
    print("last_token formation started in Decode")
    last_token_logits = last_token_logits_collection(req_id,page_table)
    print(f"last_token_logits shape {last_token_logits.shape}")
    print("last_token Completed started in Decode")
    
    generated_tokens = []
    
    with torch.no_grad():
        for _ in range(max_output_len):
    
            # 1. Convert logits → probabilities
            probs = torch.softmax(last_token_logits, dim=-1)   # (1, vocab_size)
    
            # 2. Top-k filter (k = 10 here)
            topk_probs, topk_idx = torch.topk(probs, k=10, dim=-1)  # both (1, 10)
    
            # 3. Normalize the top-k distribution
            topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
    
            # 4. Sample 1 token from top-k
            sampled_idx = torch.multinomial(topk_probs, num_samples=1)  # (1,1)
    
            # 5. Convert sampled index back to vocab ID
            next_token_id = topk_idx.gather(dim=-1, index=sampled_idx)  # (1,1)
    
            # 6. Decode token (for printing)
            token_int = next_token_id.item()
            token_text = tokenizer.decode([token_int], skip_special_tokens=True)
            generated_tokens.append(token_text)
    
            # 7. Feed into model for next step
            output = model(
                input_ids=next_token_id,             # MUST be (1,1)
                past_key_values=formed_KV_caches,    # KV cache from last step
                use_cache=True
            )
    
            # 8. Update next-step logits + KV cache
            last_token_logits = output.logits[:, -1, :]       # always shape (1, vocab_size)
            formed_KV_caches = output.past_key_values         # updated KV
    
    sentence = "".join(generated_tokens)
    print(sentence)

    print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] decode END {req_id}", flush=True)

    return {req_id , sentence}