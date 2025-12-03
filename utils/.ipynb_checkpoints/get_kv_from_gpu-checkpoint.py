# utils/get_kv_from_gpu.py    OR inside decode_worker

import torch
import math

def get_kv_from_gpu_blocks(req_id, layer_id, page_table, gpu_kv_manager):
    entry = page_table[req_id]["gpu_blocks"][layer_id]
    k_block_ids = entry["K"]
    v_block_ids = entry["V"]

    # READ FINAL SHAPE FROM PAGE TABLE
    B     = page_table[req_id]["shape"][0]      # BATCH SIZE
    seq   = page_table[req_id]["seq_lens"][0]   # TRUE sequence len
    nh_kv = page_table[req_id]["num_kv_heads"]  # you must STORE this during init
    hd    = page_table[req_id]["head_dim"]      # must be stored too!

    total_kv_elements = B * nh_kv * seq * hd

    # 1D BUFFER — Correct size
    K_full = torch.empty(total_kv_elements, dtype=torch.float16, device="cuda:0")
    V_full = torch.empty(total_kv_elements, dtype=torch.float16, device="cuda:0")

    # COPY IN CHUNKS
    k_ptr = 0
    for bid in k_block_ids:
        block = gpu_kv_manager.gpu_block_pool[bid]
        n = gpu_kv_manager.elements_in_gpu_block[bid]
        K_full[k_ptr:k_ptr+n].copy_(block[:n], non_blocking=True)
        k_ptr += n

    v_ptr = 0
    for bid in v_block_ids:
        block = gpu_kv_manager.gpu_block_pool[bid]
        n = gpu_kv_manager.elements_in_gpu_block[bid]
        V_full[v_ptr:v_ptr+n].copy_(block[:n], non_blocking=True)
        v_ptr += n

    # 🔥 CRUCIAL FINAL RESHAPE
    K_full = K_full.view(B, nh_kv, seq, hd)
    V_full = V_full.view(B, nh_kv, seq, hd)

    return K_full, V_full