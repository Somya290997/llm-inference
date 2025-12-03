import time
import torch
from datetime import datetime

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
    
    return torch.multinomial(probs, num_samples=1)

def compute_request_metrics(page_table, req_id):
    pt = page_table[req_id]   # shortcut

    # ---- RAW STAGE DURATIONS ----
    total_time         = pt["req_id_end_time"]         - pt["req_id_start_time"]
    prefill_time       = pt["prefill_end_time"]        - pt["prefill_start_time"]
    cpu_transfer_time  = pt["CPU_transfer_end_time"]   - pt["CPU_transfer_start_time"]
    gpu_transfer_time  = pt["GPU0_transfer_end_time"]  - pt["GPU0_transfer_start_time"]
    decode_time        = pt["Decode_end_time"]         - pt["Decode_start_time"]

    # Replace None or 0.0 values with 0
    total_time        = max(total_time, 0.0)
    prefill_time      = max(prefill_time, 0.0)
    cpu_transfer_time = max(cpu_transfer_time, 0.0)
    gpu_transfer_time = max(gpu_transfer_time, 0.0)
    decode_time       = max(decode_time, 0.0)

    # ---- OVERLAPS ----
    overlap_prefill_cpu = max(0.0, pt["prefill_end_time"] - pt["CPU_transfer_start_time"])
    overlap_cpu_gpu     = max(0.0, pt["CPU_transfer_end_time"] - pt["GPU0_transfer_start_time"])
    overlap_gpu_decode  = max(0.0, pt["GPU0_transfer_end_time"] - pt["Decode_start_time"] )

    return {
        "total_ms"         : total_time * 1000,
        "total_prefill_ms"       : prefill_time * 1000,
        "total_cpu_transfer_ms"  : cpu_transfer_time * 1000,
        "total_gpu_transfer_ms"  : gpu_transfer_time * 1000,
        "total_decode_ms"        : decode_time * 1000,
        "overlap_prefill_cpu_ms" : overlap_prefill_cpu * 1000,
        "overlap_cpu_gpu_ms"     : overlap_cpu_gpu * 1000,
        "overlap_gpu_decode_ms"  : overlap_gpu_decode * 1000,
    }

def decode_stage(req_id, past_key_values, logits, page_table, cpu_kv_manager, runtime):
    """
    req_id: single request_id for the entire batch.
    past_key_values: batched KV cache for all sequences.
    logits: batched logits from prefill.
    """

   
    torch.cuda.set_device(0)
    model = runtime.decode_model
    tokenizer = runtime.tokenizer

    eos = tokenizer.eos_token_id
    max_new_tokens = 200
    temperature = 0.8
    top_p = 0.9

    # -------------------------
    # BATCH SIZE
    # -------------------------
    batch_size = logits.shape[0]

    entry = page_table[req_id]

    # -------------------------
    # Decode Start Time
    # -------------------------
    if entry["Decode_start_time"] == 0.0:
        entry["Decode_start_time"] = time.time()

    # -------------------------
    # FIRST TOKEN
    # -------------------------
    next_token = sample_top_p(logits, temperature, top_p)
    page_table.global_metrics["request_count"] += batch_size
    print(f"Time to First Token is {(time.time() - entry['req_id_start_time']) * 1000:.2f} ms")
    generated_ids = next_token.clone()

    # TTFB
    ttfb_time = time.time()
    ttft_ms = (ttfb_time - entry['req_id_start_time']) * 1000

    past_kv = past_key_values
    finished = torch.zeros(batch_size, dtype=torch.bool, device=logits.device)

    tbt_times = []
    total_generated = 1

    # -------------------------
    # AUTOREGRESSIVE DECODE LOOP
    # -------------------------
    for step_idx in range(max_new_tokens - 1):

        step_start = time.time()

        with torch.no_grad():
            outputs = model(
                input_ids=next_token,
                past_key_values=past_kv,
                use_cache=True
            )

        past_kv = outputs.past_key_values
        logits = outputs.logits[:, -1, :]

        next_token = sample_top_p(logits, temperature, top_p)
        generated_ids = torch.cat([generated_ids, next_token], dim=-1)

        eos_mask = next_token.squeeze() == eos
        finished |= eos_mask
        total_generated += (1*batch_size)

        # TBT (batch-level)
        tbt_ms = (time.time() - step_start) * 1000
        tbt_times.append(tbt_ms)

        if finished.all():
            break

    # -------------------------
    # FINAL METRICS (batch-level)
    # -------------------------
    end_time = time.time()

    avg_tbt = sum(tbt_times) / len(tbt_times) if tbt_times else 0.0
    decode_ms = (end_time - entry["Decode_start_time"]) * 1000
    prefill_ms = (entry["prefill_end_time"] - entry["prefill_start_time"]) * 1000
    total_latency_ms = (end_time - entry["req_id_start_time"]) * 1000

    # -------------------------
    # UPDATE METRICS (ONCE per batch)
    # -------------------------
    entry["Decode_end_time"] = end_time
    entry["req_id_end_time"] = end_time
    entry["generated_tokens"] = total_generated

    # GLOBAL METRICS
    page_table.global_metrics["ttft_list"].append(ttft_ms)
    page_table.global_metrics["tbt_list"].append(avg_tbt)
    page_table.global_metrics["decode_ms"].append(decode_ms)
    page_table.global_metrics["prefill_ms"].append(prefill_ms)
    page_table.global_metrics["total_latency_ms"].append(total_latency_ms)

    page_table.global_metrics["generated_tokens_total"] += total_generated
    page_table.global_metrics["decode_times_total"] += (end_time - entry["Decode_start_time"])
    page_table.global_metrics["requested_completed_with_decode"] += batch_size

    # -------------------------
    # DECODE TEXTS FOR ALL SEQUENCES
    # -------------------------
    # decoded_texts = [tokenizer.decode(generated_ids[i], skip_special_tokens=True)
    #                  for i in range(batch_size)]

    # return decoded_texts
    
            
    print("completed request")

    metrics = compute_request_metrics(page_table,req_id)
    print("====== REQUEST METRICS ======")
    for k, v in metrics.items():
        print(f"{k}: {v:.2f} ms")
    print("================================\n")

            # final_text = tokenizer.decode(req.generated_ids[0], skip_special_tokens=True)
            # print(final_text)

            

            
    
    