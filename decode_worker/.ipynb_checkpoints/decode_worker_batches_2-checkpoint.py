import time
import torch
from datetime import datetime

DEBUG_DECODE = False
decode_device = "cuda:0"

_model = None
_tokenizer = None

def sample_top_p_fast(logits, temperature=0.8, top_p=0.9):
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)

    sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    cutoff = cumulative > top_p
    cutoff[..., 1:] = cutoff[..., :-1].clone()
    cutoff[..., 0] = False

    sorted_probs = torch.where(cutoff, torch.zeros_like(sorted_probs), sorted_probs)
    probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

    # pure GPU multinomial — no Python ops
    return torch.multinomial(probs, 1)


def sample_top_p(logits, temperature=0.8, top_p=0.9):
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    mask = cumulative > top_p
    mask[..., 1:] = mask[..., :-1].clone()
    mask[..., 0] = False
    
    probs[mask.scatter(1, sorted_idx, mask)] = 0.0
    probs = probs / probs.sum(dim=-1, keepdim=True)
    
    return torch.multinomial(probs, num_samples=1)  # [B,1]

def pad_kv_tensor(t, target_T):
    """Pad KV tensor to target seq_len T so we can batch safely."""
    B, H, T, D = t.shape
    if T == target_T:
        return t
    pad_len = target_T - T
    pad = torch.zeros((B, H, pad_len, D), dtype=t.dtype, device=t.device)
    return torch.cat([t, pad], dim=2)  # concat on seq_len axis


def merge_past_kv(requests):
    """
    Merge multiple single-request KV caches into batched KV format:
    returns: list of layer_idx → (batch_key, batch_value)
    """
    num_layers = len(requests[0].past_kv)
    merged = []

    for layer_idx in range(num_layers):
        k_list, v_list = [], []

        # 🔍 find MAX current sequence length across this batch
        max_T = max(req.past_kv[layer_idx][0].shape[2] for req in requests)

        for req in requests:
            k, v = req.past_kv[layer_idx]

            # ⛑️ SAFELY pad to max_T
            k = pad_kv_tensor(k, max_T)
            v = pad_kv_tensor(v, max_T)

            k_list.append(k)
            v_list.append(v)

        batch_k = torch.cat(k_list, dim=0)  # (B, H, T, D)
        batch_v = torch.cat(v_list, dim=0)
        merged.append((batch_k, batch_v))

    return merged

def decode_one_token_step(decode_batch,page_table,runtime):
    
    torch.cuda.set_device(0)
    model = runtime.decode_model
    tokenizer = runtime.tokenizer

    input_ids = torch.cat([req.generated_ids[:,-1:] for req in decode_batch], dim=0)
    past_kv = merge_past_kv(decode_batch)

    step_start = time.time()
    
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            past_key_values=past_kv,
            use_cache=True
        ) 

    # print(f"size of KV : in decode {outputs.logits[:, -1, :].shape}")

    next_logits = outputs.logits[:, -1, :]
    next_token = sample_top_p(next_logits)   # [B,1]

    tbt_ms = (time.time() - step_start) * 1000
    new_past_kvs = outputs.past_key_values
    eos = tokenizer.eos_token_id

    for i , req in enumerate(decode_batch):
    
            
        req.generated_ids = torch.cat([req.generated_ids, next_token[i].unsqueeze(0)], dim=1)
        req.total_generated += 1
        req.tbt_times.append(tbt_ms)
        
        req.past_kv = [
            (k[i:i+1], v[i:i+1]) for k, v in new_past_kvs
        ]

        if next_token[i].item() == eos or req.total_generated >= 200 :
            req.finished = True
            end_time = time.time()

            page_table[req.req_id]["Decode_start_time"] = req.decode_start_time
            page_table[req.req_id]["Decode_end_time"] = end_time
            page_table[req.req_id]["req_id_end_time"] = end_time
            page_table[req.req_id]["generated_tokens"] = req.total_generated

            # === calculate per-request averages ===
            avg_tbt = sum(req.tbt_times) / len(req.tbt_times)
            ttft_ms = (req.first_token_time - page_table[req.req_id]["req_id_start_time"]) * 1000
            total_ms = (end_time - page_table[req.req_id]["req_id_start_time"]) * 1000
            decode_ms = (end_time - req.decode_start_time) * 1000
            prefill_ms = (page_table[req.req_id]["prefill_end_time"] - page_table[req.req_id]["prefill_start_time"]) * 1000
        
            # === STORE IN GLOBAL COLLECTORS ===
            page_table.global_metrics["ttft_list"].append(ttft_ms)
            page_table.global_metrics["tbt_list"].append(avg_tbt)
            page_table.global_metrics["decode_ms"].append(decode_ms)
            page_table.global_metrics["prefill_ms"].append(prefill_ms)
            page_table.global_metrics["total_latency_ms"].append(total_ms)

            page_table.global_metrics["generated_tokens_total"] += req.total_generated
            page_table.global_metrics["decode_times_total"] += (end_time - req.decode_start_time)
            page_table.global_metrics["requested_completed_with_decode"] += 1
            
            print("completed request")

            # final_text = tokenizer.decode(req.generated_ids[0], skip_special_tokens=True)
            # print(final_text)

            

            
    
    