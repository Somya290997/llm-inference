import torch
import torch.nn as nn
import math

class PagedSdpaAttention(nn.Module):
    def __init__(self, orig_attn, page_table, gpu_kv_manager):
        super().__init__()
        self.orig = orig_attn                
        self.page_table = page_table
        self.gpu_kv_manager = gpu_kv_manager

        # copy important attributes
        self.num_heads = orig_attn.num_heads
        self.num_kv_heads = orig_attn.num_kv_heads
        self.head_dim = orig_attn.head_dim
        self.hidden_size = orig_attn.hidden_size
        
        self.q_proj = orig_attn.q_proj       
        self.k_proj = orig_attn.k_proj       
        self.v_proj = orig_attn.v_proj       
        self.out_proj = orig_attn.o_proj

        self.rotary_emb = orig_attn.rotary_emb 
    
    def forward(self, hidden_states, req_id, layer_id):
        
        # 1. Compute Q normally
        # hidden_states: (B, S, H)
        query = self.q_proj(hidden_states)
        B, S, D = query.shape

        # reshape to (B, num_heads, seq, head_dim)
        query = query.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # 2. Apply rotary embeddings (must match original behavior)
        cos, sin = self.rotary_emb(query, seq_len=S)  # same API as original
        query = self.rotary_emb.apply_rotary_pos_emb(query, cos, sin)

        # 3. Get K/V from GPU blocks (already tested in previous step)
        from utils.get_kv_from_gpu import get_kv_from_gpu_blocks
        K, V = get_kv_from_gpu_blocks(req_id, layer_id, self.page_table, self.gpu_kv_manager)
        # Shape: (B, num_kv_heads, S, head_dim)

        # 4. Expand K/V if num_kv_heads < num_heads (GQA) — REPLICATE heads
        if self.num_kv_heads != self.num_heads:
            factor = self.num_heads // self.num_kv_heads
            K = K.repeat_interleave(factor, dim=1)   # (B, num_heads, S, Hd)
            V = V.repeat_interleave(factor, dim=1)

        # 5. Compute scaled dot-product attention (manual)
        scores = torch.matmul(query, K.transpose(-1, -2)) / math.sqrt(self.head_dim)
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)  # (B, num_heads, S, Hd)

        # 6. Merge heads back & output projection
        context = context.transpose(1, 2).contiguous().view(B, S, D)
        out = self.out_proj(context)   # (B, S, hidden_size)

        return out
        