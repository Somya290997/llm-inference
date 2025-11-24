import time
import torch
from datetime import datetime
from paged_attention.paged_sdpa_attn import PagedSdpaAttention
from models.mistral_paged_wrapper import MistralPagedWrapper

DEBUG_DECODE = False
decode_device = "cuda:0"

_model = None
_tokenizer = None


def decode_stage(req_id, page_table, gpu_kv_manager, runtime, max_new_tokens=30):
    torch.cuda.set_device(0)
    
    tokenizer = runtime.tokenizer
    model = runtime.decode_model               # the original HF model
    
    # 1. Replace attention in every layer
    for layer in model.model.layers:
        layer.self_attn = PagedSdpaAttention(
            layer.self_attn,
            page_table,
            gpu_kv_manager
        )

    # 2. Use our wrapper model (for forward pass)
    paged_model = MistralPagedWrapper(model.model, page_table)

    # 3. Start decode with EOS or last prompt token
    input_ids = torch.tensor([[tokenizer.eos_token_id]], device=decode_device)

    generated_tokens = []

    # 4. REAL PAGED ATTENTION LOOP (no past_key_values)
    for _ in range(max_new_tokens):
        logits = paged_model(input_ids, req_id)   # NEW forward()
        next_token = torch.argmax(logits[:, -1, :], dim=-1)  # greedy decode
        
        generated_tokens.append(next_token.item())
        input_ids = next_token.unsqueeze(0)  # feed back as next input

    # 5. Decode tokens into text
    output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return output_text
        