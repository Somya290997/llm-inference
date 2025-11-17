import yaml
import datetime
from transformers.cache_utils import DynamicCache
import time
import torch

TRANSFER_DEBUG = True

decode_device = "cuda:0"
prefill_device = "cuda:1"

def transfer_stage(req_id, start_layers , end_layers, page_table , cpu_kv_manager):
    
    print(f"[Transfer] stage for req {req_id} has been started") if TRANSFER_DEBUG else None 

    seq_len = page_table[req_id]["seq_len"]

    with open("config/model_config.yaml", "r") as f:
        config = yaml.safe_load(f)


    hidden_dim = config["n_dim"]
    n_heads = config["n_heads"]
    batch_size = 1

    shape = (batch_size, n_heads, seq_len, hidden_dim) 

    past_key_values = DynamicCache()

    for layer_id in range(start_layers,end_layers):

        start_t = time.time()

        k_tensor , v_tensor = page_table.get_kv_gpu(req_id=req_id, layer=layer_id , shape=shape , device=decode_device ,cpu_kv_manager=cpu_kv_manager) 

        # k_hf = page_table[req_id]["hf_kv"][layer_id]["k"].to(k_tensor.device).clone()
        # v_hf = page_table[req_id]["hf_kv"][layer_id]["v"].to(v_tensor.device).clone()
        
        # # compare
        # close_k = torch.allclose(k_tensor, k_hf, atol=1e-5, rtol=1e-4)
        # diff_k = (k_tensor - k_hf).abs().max().item()
        
        # close_v = torch.allclose(v_tensor, v_hf, atol=1e-5, rtol=1e-4)
        # diff_v = (v_tensor - v_hf).abs().max().item()
        
        # print(f"[Compare][Layer {layer_id}] K close={close_k} diff={diff_k}")
        # print(f"[Compare][Layer {layer_id}] V close={close_v} diff={diff_v}")

        if layer_id == 1 and TRANSFER_DEBUG:
            print(f"[Transfer] the layer 1 shape : {k_tensor.shape} ")

        end_t = time.time()
        layers_ms = (end_t - start_t)*1000
        page_table.update_layers_at_transfer(req_id,layers_ms)

        if layer_id == 0 and TRANSFER_DEBUG :  
            print(f"[Transfer] Layer {layer_id} copy: slice={k_tensor.flatten()[:10]}")

        past_key_values.update(k_tensor,v_tensor,layer_id,cache_kwargs=None)
    
    print(f"[Transfer] stage for req {req_id} has been completed") if TRANSFER_DEBUG else None 
    
    return past_key_values
