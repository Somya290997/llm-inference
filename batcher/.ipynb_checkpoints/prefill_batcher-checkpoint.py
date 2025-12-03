import torch
from models import model_loader
import yaml
import queue
import time

_global_req_id = 0
decode_device = "cuda:0"
prefill_device = "cuda:1"
_tokenizer = None

with open("config/model_config.yaml", "r") as f:
    config = yaml.safe_load(f)


num_layers = config["num_layers"]
n_head = config["n_heads"]
n_dim = config["n_dim"]
PREFILL_BATCHER = True

def collect_batch(user_input_queue, max_batch=16, max_wait_ms=2000, timeout=2):
    curr = []
    start_time = time.time()

    while True:
        try:
            prompt = user_input_queue.get(timeout=timeout)   # wait only 0.5s
            curr.append(prompt)
        except queue.Empty:
            pass  # No request → just check time/batch

        elapsed_ms = (time.time() - start_time) * 1000

        # ---- BREAK CONDITIONS ----
        if len(curr) >= max_batch:   # Batch full
            break

        if elapsed_ms >= max_wait_ms and len(curr) > 0:  # Time exceeded
            break

    return curr

def prefill_batches_stage(user_input_queue, page_table ,batch_size, runtime):
    global _tokenizer , _global_req_id

    print(f"[PREFILL_BATCHER] Inside it ") if PREFILL_BATCHER else None

    # if _tokenizer is None:
    #     _tokenizer = model_loader.get_tokenizer()
    #     if _tokenizer.pad_token is None:
    #         _tokenizer.pad_token = _tokenizer.eos_token

    _tokenizer = runtime.tokenizer
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token
    
    req_id = _global_req_id
    _global_req_id += 1

    curr_request = collect_batch(user_input_queue, max_batch=batch_size,  max_wait_ms=2000, timeout=0.5)
    batch_size = len(curr_request)

    if batch_size == 0:
        return None, None, None
    
    curr_shape = (batch_size , n_head , 1 , n_dim)

    print(f"[PREFILL_BATCHER] for req {req_id} has started the Page_table_intialization ") if PREFILL_BATCHER else None
    page_table.init_request(req_id = req_id, batch_size = batch_size , num_layers = num_layers , seq_len = None, shape = curr_shape, dtype = torch.float16)
    print(f"[PREFILL_BATCHER] for req {req_id} has complete the Page_table_intialization  ") if PREFILL_BATCHER else None

    seq_lens_per_prompt = [len(_tokenizer.encode(prompt)) for prompt in curr_request]
    page_table[req_id]["seq_lens"] = seq_lens_per_prompt
    print(seq_lens_per_prompt)
    tokenized_output = _tokenizer(curr_request , return_tensors="pt", padding=True ).to(prefill_device)
    input_ids = tokenized_output["input_ids"]

    print(f"[PREFILL_BATCHER] for req {req_id} has input_id shape of {input_ids.shape}") if PREFILL_BATCHER else None

    return req_id , input_ids , curr_request
