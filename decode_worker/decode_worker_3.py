import time
import torch
from datetime import datetime

DEBUG_DECODE = True
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


# input_ids

# def decode_stage(req_id, past_key_values, page_table, cpu_kv_manager):

#     torch.cuda.set_device(0)
#     model, tokenizer = _load_model_once()

#     # move input_ids to GPU
#     input_ids = page_table[req_id]["input_ids"].to(decode_device)

#     eos = tokenizer.eos_token_id
#     print("EOS token =", eos)

#     last_token = input_ids[:, -1:].to(decode_device)
#     generated_ids = []
#     max_new_tokens = 500

#     decode_start = datetime.now()
#     print(f"[{decode_start.strftime('%H:%M:%S.%f')[:-3]}] Decode START {req_id}")

#     # ---------------- TTFB ----------------
#     with torch.no_grad():
#         outputs = model(
#             input_ids=last_token,
#             past_key_values=past_key_values,
#             use_cache=True
#         )

#     logits = outputs.logits[:, -1, :]

#     # deterministic first token
#     next_token = torch.argmax(logits, dim=-1, keepdim=True)

#     first_id = next_token.item()
#     generated_ids.append(first_id)

    
#     ttfb_end = datetime.now()
#     ttfb_ms = (ttfb_end - decode_start).total_seconds() * 1000
#     print(f"[Decode] TTFB for {req_id}: {ttfb_ms:.2f} ms")
#     # print(f"[Decode] TTFB token = {first_id}")

#     past_kv = outputs.past_key_values

#     if first_id == eos:
#         print("⛔ Hit EOS at TTFB — stopping.")
#         text = tokenizer.decode(generated_ids, skip_special_tokens=True)
#         print(f"[Decode] Output: {text}")
#         return

#     # ------------- AUTOREGRESSIVE LOOP -------------
#     tbt_times = []

#     for _ in range(max_new_tokens):

#         step_start = time.time()

#         with torch.no_grad():
#             outputs = model(
#                 input_ids=next_token,
#                 past_key_values=past_kv,
#                 use_cache=True
#             )

#         logits = outputs.logits[:, -1, :]
#         next_token = torch.argmax(logits, dim=-1, keepdim=True)

#         tid = next_token.item()
#         generated_ids.append(tid)

#         if tid == eos:
#             print("⛔ Hit EOS — stopping generation.")
#             break

#         past_kv = outputs.past_key_values
#         tbt_times.append((time.time() - step_start) * 1000)

#     # ---------------- FINISH ----------------
#     decode_end = datetime.now()
    
#     avg_tbt = sum(tbt_times)/len(tbt_times) if tbt_times else 0.0

#     text = tokenizer.decode(generated_ids, skip_special_tokens=True)

#     print(f"[{decode_end.strftime('%H:%M:%S.%f')[:-3]}] Decode END {req_id}")
#     print(f"[Decode] AVG TBT = {avg_tbt:.2f} ms over {len(tbt_times)} tokens")
#     print(f"[Decode] Output:\n{text}")



def decode_stage(req_id, past_key_values, page_table, cpu_kv_manager):

    torch.cuda.set_device(0)
    model, tokenizer = _load_model_once()

    eos = tokenizer.eos_token_id
    generated_ids = []
    max_new_tokens = 200

    decode_start = datetime.now()
    print(f"[{decode_start.strftime('%H:%M:%S.%f')[:-3]}] Decode START {req_id}")

    # --------------------------------------------------------
    # 1) FIRST TOKEN — FROM PREFILL LOGITS (NO MODEL CALL)
    # --------------------------------------------------------
    # choose token from the prefill logits
    last_logits = page_table[req_id]["last_layer_logits"]
    next_token = torch.argmax(last_logits, dim=-1, keepdim=True)
    token_id = next_token.item()
    generated_ids.append(token_id)

    ttfb_end = datetime.now()
    ttfb_ms = (ttfb_end - decode_start).total_seconds() * 1000
    print(f"[Decode] TTFB for {req_id}: {ttfb_ms:.2f} ms")

    print(f"[Decode] First token = {token_id}")

    # EOS check
    if token_id == eos:
        print("EOS from prefill — stopping.")
        print(tokenizer.decode(generated_ids, skip_special_tokens=True))
        return

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
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        token_id = next_token.item()
        generated_ids.append(token_id)

        if token_id == eos:
            print("EOS — stopping generation.")
            break

        past_kv = outputs.past_key_values

        tbt_times.append((time.time() - step_start) * 1000)

    # --------------------------------------------------------
    # 3) FINAL OUTPUT
    # --------------------------------------------------------
    avg_tbt = sum(tbt_times)/len(tbt_times) if tbt_times else 0.0
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    print("---- DECODE END ----")
    print(f"[Decode] AVG TBT = {avg_tbt:.2f} ms over {len(tbt_times)} tokens")
    print(f"[Decode] Output:\n{text}")