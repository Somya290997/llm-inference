# utils/get_kv_from_gpu.py    OR inside decode_worker

import torch
import math

def get_kv_from_gpu_blocks(req_id, layer_id, page_table, gpu_kv_manager):
    """
    Reconstruct K and V for a single layer using only GPU blocks.
    Returns tensors shaped:
      K: (B, num_kv_heads, seq_len, head_dim)
      V: same
    """
    entry = page_table[req_id]
    gpu_layer_info = entry["gpu_blocks"][layer_id]

    # Extract shape metadata
    B   = entry["shape"][0]
    S   = entry["shape"][2]              # total seq_len
    Hd  = entry["head_dim"]
    Hkv = entry["num_kv_heads"]

    # allocate empty full tensors
    K_full = torch.empty(B * Hkv * S * Hd, device="cuda:0", dtype=torch.float16)
    V_full = torch.empty_like(K_full)

    # Reconstruct K
    k_ptr = 0
    for block_id in gpu_layer_info["K"]:
        block = gpu_kv_manager.get_block(block_id)   # (1D tensor)
        n = block.numel()
        K_full[k_ptr : k_ptr + n].copy_(block[:n], non_blocking=True)
        k_ptr += n

    # Reconstruct V
    v_ptr = 0
    for block_id in gpu_layer_info["V"]:
        block = gpu_kv_manager.get_block(block_id)
        n = block.numel()
        V_full[v_ptr : v_ptr + n].copy_(block[:n], non_blocking=True)
        v_ptr += n

    # reshape correctly
    K_full = K_full.view(B, Hkv, S, Hd)
    V_full = V_full.view(B, Hkv, S, Hd)
    return K_full, V_full