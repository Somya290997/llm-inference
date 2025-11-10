import torch
from datetime import datetime
from transformers import AutoTokenizer
from transformers.cache_utils import DynamicCache

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


def decode_stage(req_id, page_table):

    torch.cuda.set_device(1)
    model , tokenizer = _load_model_once()

    entry = page_table.table[req_id]
    num_layers = entry["num_layers"]
    seq_len = entry["seq_len"]
    # cpu_layers = entry["cpu_kv"]

    print(f"[Decode] START for req {req_id}")

    gpu_kv = {}

    cpu_layers = page_table[req_id]["cpu_kv"]

    past_key_values = DynamicCache()

    for layer_id in range(num_layers):
    
        # ✅ FIXED
        k_cpu = cpu_layers[layer_id]["K"]
        v_cpu = cpu_layers[layer_id]["V"]
    
        # Now both are CPU tensors
        Kgpu = torch.empty_like(k_cpu, device="cuda:0")
        Vgpu = torch.empty_like(v_cpu, device="cuda:0")
    
        Kgpu.copy_(k_cpu, non_blocking=True)
        Vgpu.copy_(v_cpu, non_blocking=True)

        if layer_id == 0 :  
            print(f"[Decode] Layer0 copy: slice={Kgpu.flatten()[:10]}")

        past_key_values.update(Kgpu,Vgpu,layer_id,cache_kwargs=None)
    
        # optionally store GPU KV for decoder
        # page_table.set_gpu_kv(req_id, layer_id, Kgpu, Vgpu)
    


    print(f"[Decode] All KV copied. Starting token generation...")

    # ----- simple generate -----
    last_token_logits = entry["logits_cpu"]
    # next_token = torch.argmax(logits_cpu, dim=-1) 

    generated_tokens = []
    max_output_len = 120
    
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
                past_key_values=past_key_values,    # KV cache from last step
                use_cache=True
            )
    
            # 8. Update next-step logits + KV cache
            last_token_logits = output.logits[:, -1, :]       # always shape (1, vocab_size)
            formed_KV_caches = output.past_key_values         # updated KV
    
    sentence = "".join(generated_tokens)
    print(sentence)

    print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] decode END {req_id}", flush=True)

    # print(f"[Decode] first token = {next_token}")
    # print(f"[Decode] DONE")





