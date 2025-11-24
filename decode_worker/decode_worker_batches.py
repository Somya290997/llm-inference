import time
import torch
from datetime import datetime

DEBUG_DECODE = False
decode_device = "cuda:0"

_model = None
_tokenizer = None


def _load_model_once():
    """Load model ONCE globally."""
    global _model, _tokenizer

    if _model is None:
        from models import model_loader
        print("[Decode] Loading model on cuda:0...")
        _model = model_loader.get_model("cuda:0")
        _tokenizer = model_loader.get_tokenizer()
        print("[Decode] Model loaded ✓")

    return _model, _tokenizer


def compute_request_metrics(page_table, req_id):
    pt = page_table[req_id]

    total_time         = pt["req_id_end_time"]         - pt["req_id_start_time"]
    prefill_time       = pt["prefill_end_time"]        - pt["prefill_start_time"]
    cpu_transfer_time  = pt["CPU_transfer_end_time"]   - pt["CPU_transfer_start_time"]
    gpu_transfer_time  = pt["GPU0_transfer_end_time"]  - pt["GPU0_transfer_start_time"]
    decode_time        = pt["Decode_end_time"]         - pt["Decode_start_time"]

    total_time        = max(total_time, 0.0)
    prefill_time      = max(prefill_time, 0.0)
    cpu_transfer_time = max(cpu_transfer_time, 0.0)
    gpu_transfer_time = max(gpu_transfer_time, 0.0)
    decode_time       = max(decode_time, 0.0)

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
    

def decode_stage(req_id, past_key_values, logits, page_table, cpu_kv_manager, runtime):
    torch.cuda.set_device(0)
    model = runtime.decode_model
    tokenizer = runtime.tokenizer

    eos = tokenizer.eos_token_id
    max_new_tokens = 128
    temperature = 0.8
    top_p = 0.9

    # ---- Track batch size ----
    batch_size = logits.shape[0]  # e.g. 5
    print(f"\n[Decode] Starting batched decode for {batch_size} requests")

    if page_table[req_id]["Decode_start_time"] == 0.0:
        page_table[req_id]["Decode_start_time"] = time.time()

    decode_start_time = time.time()
    
    # ---- INITIAL TOKEN SELECTION (FIRST TOKEN) ----
    next_token = sample_top_p(logits, temperature, top_p)   # [B,1]
    generated_ids = next_token.clone()                      # store full sequences

    # ⏱️ TTFB: Time to generate first token after prefill
    ttfb_end_time = time.time()
    ttfb_ms = (ttfb_end_time - page_table[req_id]["prefill_start_time"]) * 1000

    past_kv = past_key_values  # KV from prefill
    finished = torch.zeros(batch_size, dtype=torch.bool, device=logits.device)

    tbt_times = []

    # --------------------------------------------------------
    #   AUTOREGRESSIVE LOOP (BATCHED)
    # --------------------------------------------------------

    total_token_generated = 0
    
    for step_idx in range(max_new_tokens - 1):  # -1 because first token already generated
        step_start = time.time()

        with torch.no_grad():
            outputs = model(
                input_ids=next_token,       # [B,1]
                past_key_values=past_kv,
                use_cache=True
            )

        # update KV
        past_kv = outputs.past_key_values

        # get logits for the last token
        logits = outputs.logits[:, -1, :]     # [B, vocab]

        # sample next token batched
        next_token = sample_top_p(logits, temperature, top_p)  # [B,1]

        # append to generated sequence
        generated_ids = torch.cat([generated_ids, next_token], dim=-1)

        # check EOS per request
        eos_mask = next_token.squeeze() == eos
        finished |= eos_mask
        if finished.all():
            print(f"[Decode] All requests finished at step {step_idx} (EOS reached)")
            break

        # compute TBT for this step
        step_time_ms = (time.time() - step_start) * 1000
        tbt_times.append(step_time_ms)
        total_token_generated = step_idx + 1

    # --------------------------------------------------------
    #   FINAL DECODE & METRICS
    # --------------------------------------------------------
    end_time = time.time()
    avg_tbt = sum(tbt_times) / len(tbt_times) if tbt_times else 0.0
    
    # update page_table
    if page_table[req_id]["Decode_end_time"] == 0.0:
        page_table[req_id]["Decode_end_time"] = end_time
        page_table[req_id]["req_id_end_time"] = end_time

    page_table[req_id]["generated_tokens"] = total_token_generated
    
    # ========== PRINT GENERATION METRICS ==========
    print(f"\n{'='*60}")
    print(f"DECODE PERFORMANCE METRICS")
    print(f"{'='*60}")
    print(f"⏱️  TTFB (Time to First Token):    {ttfb_ms:.2f} ms")
    print(f"⏱️  Average TBT (Token-by-Token):  {avg_tbt:.2f} ms")
    print(f"📊 TBT samples collected:          {len(tbt_times)}")
    print(f"📝 Tokens per request:             {generated_ids.shape[1]}")
    print(f"{'='*60}\n")

    # Decode all generated texts
    decoded_texts = []
    for i in range(batch_size):
        text = tokenizer.decode(generated_ids[i], skip_special_tokens=True)
        decoded_texts.append(text)
        # print(f"[Decode] Request {i}: {text}")

    # ========== PRINT FULL REQUEST METRICS ==========
    print(f"\n{'='*60}")
    print(f"FULL REQUEST METRICS")
    print(f"{'='*60}")
    metrics = compute_request_metrics(page_table, req_id)
    for k, v in metrics.items():
        print(f"{k:30s}: {v:8.2f} ms")
    print(f"{'='*60}\n")
    
    return decoded_texts