import torch
from datetime import datetime
from transformers import AutoTokenizer

def _load_model_once():
    """Load model ONCE when first called"""
    global _model, _tokenizer
    
    if _model is None:
        from models import model_loader
        print(f"[Prefill] Loading model on cuda:1...")
        _model = model_loader.get_model("cuda:1")
        _tokenizer = model_loader.get_tokenizer()
        print(f"[Prefill] Model loaded ✓")
    
    return _model, _tokenizer


def decode_stage(req_id, page_table, model):

    torch.cuda.set_device(1)
    # _load_model_once()

    entry = page_table.table[req_id]
    num_layers = entry["num_layers"]
    seq_len = entry["seq_len"]
    cpu_layers = entry["cpu_kv"]

    print(f"[Decode] START for req {req_id}")

    gpu_kv = {}

    for layer_id in num_layers:
        k_cpu , v_cpu = cpu_layers[layer_id]

        Kgpu = torch.empty_like(k_cpu, device="cuda:0")
        Vgpu = torch.empty_like(v_cpu, device="cuda:0")

        Kgpu.copy_(k_cpu, non_blocking=True)
        Vgpu.copy_(v_cpu, non_blocking=True)

        gpu_kv[layer_id] = {"k": Kgpu , "V": Vgpu}

        if layer_id == 0 :  
            print(f"[Decode] Layer0 copy: slice={Kgpu.flatten()[:10]}")

    print(f"[Decode] All KV copied. Starting token generation...")

    # ----- simple generate -----
    logits_cpu = entry["logits_cpu"]
    next_token = torch.argmax(logits_cpu, dim=-1)

    print(f"[Decode] first token = {next_token}")
    print(f"[Decode] DONE")





