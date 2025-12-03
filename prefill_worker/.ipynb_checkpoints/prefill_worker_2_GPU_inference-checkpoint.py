import torch
import cupy as cp
from transformers.cache_utils import DynamicCache
from datetime import datetime
import time
import yaml

DEBUG_PREFILL = True

# Loading the Yaml files
with open("config/model_config.yaml", "r") as f:
    config = yaml.safe_load(f)

decode_device = "cuda:0"
prefill_device = "cuda:1"

_model = None
_tokenizer = None

def enable_p2p():
    for i in range(2):
        for j in range(2):
            if i != j:
                with cp.cuda.Device(i):
                    try:
                        cp.cuda.runtime.deviceEnablePeerAccess(j)
                    except:
                        pass

def _load_model_once():
    """Load model ONCE when first called"""
    global _model, _tokenizer
    
    if _model is None:
        from models import model_loader
        print(f"[Prefill] Loading model on cuda:1...") if DEBUG_PREFILL else None
        _model = model_loader.get_model("cuda:1")
        _tokenizer = model_loader.get_tokenizer()
        print(f"[Prefill] Model loaded ✓") if DEBUG_PREFILL else None
    
    return _model, _tokenizer

def prefill_stage(req_id, input_ids , request_prompt , page_table,cpu_kv_manager, kv_write_to_cpu_queue , runtime):

    torch.cuda.set_device(1)

    print(f"[Prefill] for req {req_id} has been started ")
    # enable_p2p()
    print(f"[Prefill] for req {req_id} has Enable P2P ") if DEBUG_PREFILL else None

    model = runtime.prefill_model
    tokenizer =  runtime.tokenizer

    seq_len = input_ids.shape[1]
    print(f"Number of tokens in {req_id} is {seq_len} ")

    print(f"Prefill START {req_id}") if DEBUG_PREFILL else None

    print(f"[Prefill] for req {req_id} has started forward pass") if DEBUG_PREFILL else None

    page_table[req_id]["prefill_start_time"] = time.time()
    
    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)


    KV_cache = outputs.past_key_values

    for layer_idx , KV in enumerate(KV_cache):
        K, V = KV
        k_clone = K.detach().contiguous()
        v_clone = V.detach().contiguous()
        kv_write_to_cpu_queue.put((req_id,layer_idx,k_clone,v_clone))
        

    print(f"[Prefill] for req {req_id} has compled streamingthe KV cache") if DEBUG_PREFILL else None
    

        

    # -------- LOGITS EXTRACTION (SAFEST VERSION) --------
    logits = outputs.logits  # [B, T, vocab]
    
    # Retrieve real sequence lengths
    seq_lens_per_prompt = page_table[req_id]["seq_lens"]  # example: [8, 15, 13, 14, 8]
    
    # Convert to torch tensor on correct device
    seq_lens_tensor = torch.tensor(seq_lens_per_prompt, device=logits.device)
    
    # --- CRITICAL: FIXING 0-D TENSOR (batch_size = 1 case) ---
    if seq_lens_tensor.ndim == 0:      # e.g., tensor(15)
        seq_lens_tensor = seq_lens_tensor.unsqueeze(0)
    
    batch_size = seq_lens_tensor.shape[0]
    print(f"[Prefill] batch_size = {batch_size}, seq_lens_tensor = {seq_lens_tensor}")
    
    # SAFE extraction of last token logits per request
    real_last_logits = logits[
        torch.arange(batch_size, device=logits.device),
        seq_lens_tensor - 1,
        :
    ]

    real_last_logits_cpu = real_last_logits.detach().contiguous()
    page_table.set_logits_kv_cpu(req_id,real_last_logits_cpu,cpu_kv_manager)
    
    print(f"[Prefill] real_last_logits shape = {real_last_logits.shape}")

    torch.cuda.synchronize(prefill_device)

    

    print(f"[Prefill] for req {req_id} has completed forward pass") if DEBUG_PREFILL else None


    page_table[req_id]["prefill_end_time"] = time.time()

    

    