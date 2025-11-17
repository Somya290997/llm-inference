import time
import torch
from datetime import datetime

DEBUG_DECODE = True

_model = None
_tokenizer = None

decode_device = "cuda:0"
prefill_device = "cuda:1"

def _load_model_once():
    """Load model ONCE when first called"""
    global _model, _tokenizer
    
    if _model is None:
        from models import model_loader
        print(f"[Decode] Loading model on cuda:0...") if DEBUG_DECODE else None
        _model = model_loader.get_model("cuda:0")
        _tokenizer = model_loader.get_tokenizer()
        print(f"[Decode] Model loaded") if DEBUG_DECODE else None
    
    return _model, _tokenizer


def decode_stage(req_id, past_key_values ,page_table, cpu_kv_manager):
    
    torch.cuda.set_device(0)
    model , tokenizer = _load_model_once()

    input_ids = page_table[req_id]["input_ids"].to(decode_device)

    last_token = input_ids[:,-1:]

    generated_tokens = []
    max_output_len = 64

    decode_start = datetime.now()
    print(f"[{decode_start.strftime('%H:%M:%S.%f')[:-3]}] Decode START {req_id}") if DEBUG_DECODE else None

    with torch.no_grad():

        outputs = model(
            input_ids = last_token,
            past_key_values=past_key_values,
            use_cache = True
        )

        logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(logits, dim=-1, keepdim=True)

        ttfb_end = datetime.now()
        ttfb_ms = (ttfb_end - decode_start).total_seconds() * 1000
        print(f"[Decode] TTFB for {req_id}: {ttfb_ms:.2f} ms")

        # decode first token
        tok_str = tokenizer.decode(next_token[0])
        generated_tokens.append(tok_str)

        past_kv = outputs.past_key_values  # update PKV

        last_token_logits = logits

        print(f"[{decode_start.strftime('%H:%M:%S.%f')[:-3]}] Decode is not between for {req_id}") if DEBUG_DECODE else None

        # ------------------------------------
        # AUTOREGRESSIVE LOOP → TBT
        # ------------------------------------
        tbt_times = []

        for _ in range(max_output_len):

            step_start = time.time()

            # convert logits to probabilities
            probs = torch.softmax(last_token_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

            # decode token to text
            tok_str = tokenizer.decode(next_id[0])
            generated_tokens.append(tok_str)

            # forward step
            with torch.no_grad():
                outputs = model(
                    input_ids=next_id,
                    past_key_values=past_kv,
                    use_cache=True
                )

            past_kv = outputs.past_key_values
            last_token_logits = outputs.logits[:, -1, :]

            # measure TBT for this token
            step_tbt = (time.time() - step_start) * 1000
            tbt_times.append(step_tbt)

        # ------------------------------------
        # FINISH AND REPORT
        # ------------------------------------
        
        print(f"[{decode_start.strftime('%H:%M:%S.%f')[:-3]}] Decode for {req_id} has been completed") if DEBUG_DECODE else None

        avg_tbt = sum(tbt_times) / len(tbt_times)

        decode_end = datetime.now()
        total_ms = (decode_end - decode_start).total_seconds() * 1000

        print(f"[{decode_end.strftime('%H:%M:%S.%f')[:-3]}] Decode END {req_id}   (Total={total_ms:.2f} ms)")
        print(f"[Decode] AVG TBT={avg_tbt:.2f} ms over {len(tbt_times)} tokens")

        # join tokens
        sentence = "".join(generated_tokens)
        print(f"[Decode] Output: {sentence}")
    







