import time
import torch
from datetime import datetime
from decode_worker.ActiveRequest_2 import BatchedActiveRequest

DEBUG_DECODE = False
decode_device = "cuda:0"

_model = None
_tokenizer = None


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

def decode_one_token_batched(batched_req, page_table, runtime):

    torch.cuda.set_device(0)
    model     = runtime.decode_model
    tokenizer = runtime.tokenizer

    input_ids = batched_req.generated_ids[:, -1:]      # (B,1)
    past_kv   = batched_req.past_kv                    # list of (B,H,T,D)

    step_start = time.time()
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            past_key_values=past_kv,
            use_cache=True
        )

    next_logits = outputs.logits[:, -1, :]                 # (B,vocab)
    next_tokens = sample_top_p(next_logits)               # (B,1)

    tbt_ms      = (time.time() - step_start) * 1000        # FOR METRICS
    new_past_kv = outputs.past_key_values                  # LIST of (B,H,T,D)
    eos         = tokenizer.eos_token_id

    # ==============================
    # UPDATE FOR EACH REQUEST IN BATCH
    # ==============================
    finished_now = []
    for i in range(batched_req.batch_size):
        req_id = batched_req.request_ids[i]   # (0,idx)

        # append token
        batched_req.generated_ids[i] = torch.cat(
            [batched_req.generated_ids[i], next_tokens[i]], dim=0
        )
        batched_req.total_generated[i] += 1
        batched_req.tbt_times[i].append(tbt_ms)

        # only keep own slice of KV
        batched_req.past_kv = [
            (k[i:i+1], v[i:i+1]) for k, v in new_past_kv
        ]

        # FINISH IF EOS or MAX TOKENS
        if (next_tokens[i].item() == eos) or (batched_req.total_generated[i] >= 200):
            finished_now.append(i)

    # ==============================
    # PROCESS FINISHED REQUESTS
    # ==============================
    if len(finished_now) > 0:

        for i in finished_now:
            req_id = batched_req.request_ids[i]  # (req_id, idx)
            end_time = time.time()

            page_table[req_id]["Decode_start_time"] = batched_req.decode_start_time
            page_table[req_id]["Decode_end_time"]   = end_time
            page_table[req_id]["req_id_end_time"]   = end_time
            page_table[req_id]["generated_tokens"]  = batched_req.total_generated[i]

            # === METRICS (SAME AS OLD) ===
            ttft_ms   = (batched_req.first_token_time[i] - page_table[req_id]["req_id_start_time"]) * 1000
            decode_ms = (end_time - batched_req.decode_start_time) * 1000
            total_ms  = (end_time - page_table[req_id]["req_id_start_time"]) * 1000
            prefill_ms = (page_table[req_id]["prefill_end_time"] - page_table[req_id]["prefill_start_time"]) * 1000
            avg_tbt   = sum(batched_req.tbt_times[i]) / max(1, len(batched_req.tbt_times[i]))

            page_table.global_metrics["ttft_list"].append(ttft_ms)
            page_table.global_metrics["tbt_list"].append(avg_tbt)
            page_table.global_metrics["decode_ms"].append(decode_ms)
            page_table.global_metrics["prefill_ms"].append(prefill_ms)
            page_table.global_metrics["total_latency_ms"].append(total_ms)

            page_table.global_metrics["generated_tokens_total"] += batched_req.total_generated[i]
            page_table.global_metrics["decode_times_total"] += (end_time - batched_req.decode_start_time)
            page_table.global_metrics["requested_completed_with_decode"] += 1

            print(f"[Completed][req {req_id}]")

    # ==============================
    # STOP WHEN ALL FINISHED
    # ==============================
    if all(batched_req.total_generated[i] >= 200 for i in range(batched_req.batch_size)):
        print(f"[Batch Done] batch {batched_req.batch_id} completed.")
        return True

    return False

            
    
    