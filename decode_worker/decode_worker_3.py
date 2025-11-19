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



def decode_stage(req_id, past_key_values,logits, page_table, cpu_kv_manager ,start_time):

    torch.cuda.set_device(0)
    model, tokenizer = _load_model_once()

    eos = tokenizer.eos_token_id
    generated_ids = []
    max_new_tokens = 128

    decode_start = datetime.now()
    print(f"[{decode_start.strftime('%H:%M:%S.%f')[:-3]}] Decode START {req_id}")

    last_logits = logits
    # next_token = torch.argmax(last_logits, dim=-1, keepdim=True)

    temperature = 0.7
    top_p = 0.9
    
    # Apply temperature
    logits = last_logits / temperature
    
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
    
    token_id = next_token.item()
    generated_ids.append(token_id)

    ttfb_end = datetime.now()
    ttfb_ms = (ttfb_end - decode_start).total_seconds() * 1000
    req_time =  (ttfb_end - start_time).total_seconds() * 1000
    print(f"[Decode] Time to first token for {req_id}: {ttfb_ms:.2f} ms and time to start token generation is {req_time:.2f} ms")

    print(f"[Decode] First token = {token_id}") if DEBUG_DECODE else None 

    # # EOS check
    # if token_id == eos:
    #     print("EOS from prefill — stopping.")
    #     print(tokenizer.decode(generated_ids, skip_special_tokens=True))
    #     return

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

        token_id = next_token.item()
        generated_ids.append(token_id)

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

    print("---- DECODE END ----")
    print(f"[Decode] AVG TBT = {avg_tbt:.2f} ms over {len(tbt_times)} tokens")
    print(f"[Decode] Output:\n{text}") 