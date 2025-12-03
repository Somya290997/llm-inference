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

# inside PageTable class
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


def decode_stage(req_id, past_key_values,logits, page_table, cpu_kv_manager):

    torch.cuda.set_device(0)
    model, tokenizer = _load_model_once()

    eos = tokenizer.eos_token_id
    generated_ids = []
    max_new_tokens = 128
    # repetition_penalty = 1.1

    if page_table[req_id]["Decode_start_time"] == 0.0:
        page_table[req_id]["Decode_start_time"] = time.time()

    # decode_start = datetime.now()
    print(f"Decode START {req_id}")

    # next_token = torch.argmax(last_logits, dim=-1, keepdim=True)

    temperature = 0.8
    top_p = 0.9
    
    # Apply temperature
    logits = logits / temperature
    # Softmax to get probabilities
    probs = torch.softmax(logits, dim=-1)
    
    # Top-p (nucleus) sampling
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    
    # Zero out low-probability tokens
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    probs[indices_to_remove] = 0.0
    probs = probs / probs.sum(dim=-1, keepdim=True)
    
    # Sample
    next_token = torch.multinomial(probs, num_samples=1)
    generated_ids.append(next_token.item())

    ttfb_end = time.time()
    # ttfb_ms = (ttfb_end - decode_start).total_seconds() * 1000
    req_time =  (ttfb_end - page_table[req_id]["req_id_start_time"]) * 1000

    # we start decode with updated KV from prefill
    past_kv = past_key_values

    # --------------------------------------------------------
    # 2) AUTOREGRESSIVE LOOP
    # --------------------------------------------------------
    tbt_times = []

    for _ in range(max_new_tokens):

        step_start = time.time()

        # normal decode step
        with torch.no_grad():
            outputs = model(
                input_ids=next_token,    # feed previous prediction
                past_key_values=past_kv,
                use_cache=True
            )

        logits = outputs.logits[:, -1, :]

        # choose next token
        logits = logits / temperature
    
        # Softmax to get probabilities
        probs = torch.softmax(logits, dim=-1)
        
        # Top-p (nucleus) sampling
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        # Zero out low-probability tokens
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        probs[indices_to_remove] = 0.0
        probs = probs / probs.sum(dim=-1, keepdim=True)
        
        # Sample
        next_token = torch.multinomial(probs, num_samples=1)
        generated_ids.append(next_token.item())

        if next_token[0].item() == eos:
            print("EOS — stopping generation.")
            break

        past_kv = outputs.past_key_values
        
        tbt_times.append((time.time() - step_start) * 1000)

    # --------------------------------------------------------
    # 3) FINAL OUTPUT
    # --------------------------------------------------------
    avg_tbt = sum(tbt_times)/len(tbt_times) if tbt_times else 0.0
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    if page_table[req_id]["Decode_end_time"] == 0.0:
        page_table[req_id]["Decode_end_time"] = time.time()
        page_table[req_id]["req_id_end_time"] = time.time()

    print("---- DECODE END ----")
    print(f"[Decode] Output:\n{text}")
    
    print("---- RESULT ----")
    print(f"[Decode] AVG TBT = {avg_tbt:.2f} ms over {len(tbt_times)} tokens")
    print(f"[Decode] Time to first token for {req_id}: {req_time:.2f} ms")
    metrics = compute_request_metrics(page_table,req_id)
    print("====== REQUEST METRICS ======")
    for k, v in metrics.items():
        print(f"{k}: {v:.2f} ms")
    print("================================\n")
    
    