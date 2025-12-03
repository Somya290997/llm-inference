import torch
import torch.nn as nn
import math
from utils.get_kv_from_gpu import get_kv_from_gpu_blocks

class PagedSdpaAttention(nn.Module):
    def __init__(self, orig_attn, page_table, gpu_kv_manager, model_config):
        super().__init__()
        self.orig = orig_attn                
        self.page_table = page_table
        self.gpu_kv_manager = gpu_kv_manager

        # copy important attributes
        self.num_heads     = model_config.num_attention_heads        # 32
        self.num_kv_heads  = model_config.num_key_value_heads        # 8
        self.hidden_size   = model_config.hidden_size                # 4096
        self.head_dim      = self.hidden_size // self.num_heads      # 128
        
        self.q_proj = orig_attn.q_proj       
        # self.k_proj = orig_attn.k_proj       
        # self.v_proj = orig_attn.v_proj       
        self.out_proj = orig_attn.o_proj
        self.o_proj = orig_attn.o_proj

        self.rotary_emb = orig_attn.rotary_emb 
    
    def forward(self, hidden_states, *args, req_id=None, layer_id=None, **kwargs):
        # -----------------------------
        # 1) Get shape of input
        # -----------------------------
        B, S, _ = hidden_states.shape  # during decode → S = 1  🔥
    
        # -----------------------------
        # 2) Compute Query (Q)
        # -----------------------------
        query = self.q_proj(hidden_states)  # (B, S, H)
    
        # reshape → (B, num_heads, S, head_dim)
        query = query.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        # shape now → (B, num_heads, S, head_dim)
    
        # -----------------------------
        # 3) Apply rotary embeddings (safe)
        # -----------------------------
        rot_out = self.rotary_emb(query, seq_len=S)
    
        if isinstance(rot_out, torch.Tensor):
            query = rot_out
        elif isinstance(rot_out, tuple):
            query = rot_out[0]   # rotated Q only
        else:
            raise RuntimeError(f"Unexpected rotary_emb return type: {type(rot_out)}")
    
        # -----------------------------
        # 4) Fetch KV from GPU blocks
        # -----------------------------
        K, V = get_kv_from_gpu_blocks(req_id, layer_id, self.page_table, self.gpu_kv_manager)
        # shapes expected: (B, num_kv_heads, seq_len, head_dim)
    
        # -----------------------------
        # 5) Expand KV heads (GQA logic)
        # -----------------------------
        if self.num_kv_heads < self.num_heads:
            factor = self.num_heads // self.num_kv_heads
            K = K.repeat_interleave(factor, dim=1)
            V = V.repeat_interleave(factor, dim=1)
        # shapes now: (B, num_heads, seq_len, head_dim)
    
        # -----------------------------
        # 6) Scaled Dot-Product Attention
        # -----------------------------
        scores = torch.matmul(query, K.transpose(-1, -2)) / math.sqrt(self.head_dim)
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)  # (B, num_heads, S, head_dim)
    
        # -----------------------------
        # 7) Merge Heads + Project Output
        # -----------------------------
        context = context.transpose(1, 2).contiguous().view(B, S, self.hidden_size)
    
        # IMPORTANT: HuggingFace expects 3 return values
        return self.o_proj(context), None, None
        