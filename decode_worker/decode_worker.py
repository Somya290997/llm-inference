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
max_seq_len = 120
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

    model, tokenizer = _load_model_once()
    print("KV formation started in Decode")
    formed_KV_caches = kv_cache_collection(req_id,num_layers,page_table)
    print("KV Completed started in Decode")
    K, V = formed_KV_caches[0]
    print(f"KV shape {K.shape} and {V.shape}")
    print("last_token formation started in Decode")
    last_token_logits = last_token_logits_collection(req_id,page_table)
    print(f"last_token_logits shape {last_token_logits.shape}")
    print("last_token Completed started in Decode")
    


    generated_words = []

    with torch.no_grad():
        for i in range(max_output_len):
            next_token_id = torch.softmax(last_token_logits,dim=-1)
            logits_val , logis_idx = torch.topk(next_token_id,10,dim=-1)
            top_k_probs = logits_val / torch.sum(logits_val,dim=-1)
            probs = torch.multinomial(top_k_probs,1)
            next_token_id = torch.gather(logis_idx, dim=-1, index=probs)
            # word = tokenizer.decode(next_token_id[0])
            word = tokenizer.decode([next_token_id.item()], skip_special_tokens=True)
            generated_words.append(word)
            next_ids_model = next_token_id.unsqueeze(0)   # (1,1)
            output_next = model(input_ids=next_ids_model, past_key_values=formed_KV_caches)
            # output_next = model(input_ids=next_token_id,past_key_values=formed_KV_caches)
            last_token_logits = output_next.logits[:,-1,:]
            formed_KV_caches = output_next.past_key_values

    sentence = " ".join(generated_words)
    print(sentence)

    print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] decode END {req_id}", flush=True)

    return {req_id , sentence}