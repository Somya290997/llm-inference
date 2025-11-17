import torch
import cupy as cp
from transformers.cache_utils import DynamicCache
from datetime import datetime
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

def prefill_stage(req_id,prompt,page_table,cpu_kv_manager,schedular_queue):

    torch.cuda.set_device(1)
    print(f"[Prefill] for req {req_id} has been started ") if DEBUG_PREFILL else None

    enable_p2p()

    print(f"[Prefill] for req {req_id} has Enable P2P ") if DEBUG_PREFILL else None

    model , tokenizer = _load_model_once()

    inputs1 = tokenizer(prompt, return_tensors="pt")
    seq_len = inputs1["input_ids"].shape[1]

  
    inputs = tokenizer(prompt, return_tensors="pt").to(prefill_device)

    print(f"Number of tokens in {req_id} is {seq_len} ") if DEBUG_PREFILL else None

    hidden_dim = model.config.hidden_size // model.config.num_attention_heads
    batch_size = 1
    kv_shape = (batch_size, model.config.num_attention_heads, seq_len, hidden_dim)

    print(f"[Prefill] for req {req_id} has started the Page_table_intialization ") if DEBUG_PREFILL else None
    page_table.init_request(req_id = req_id, num_layers=model.config.num_hidden_layers, seq_len=seq_len, shape = kv_shape, dtype=torch.float16)
    print(f"[Prefill] for req {req_id} has complete the Page_table_intialization  ") if DEBUG_PREFILL else None

    page_table[req_id]["input_ids"] = inputs1["input_ids"]
    
    prefill_start = datetime.now()
    print(f"[{prefill_start.strftime('%H:%M:%S.%f')[:-3]}] Prefill START {req_id}") if DEBUG_PREFILL else None

    class CPUStreamingCache(DynamicCache):
        def update(self, key_states, value_states, layer_idx, cache_kwargs=None):

            # Clone to keep alive beyond layer update
            k_clone = key_states.clone().detach().cpu()
            v_clone = value_states.clone().detach().cpu()

            print(f"[Prefill] for req {req_id} shape {k_clone.shape}") if DEBUG_PREFILL and layer_idx == 0 else None 

            print(f"[Prefill] for req {req_id} has started for Set_KV_CPU for {layer_idx}") if DEBUG_PREFILL and layer_idx == 0 else None 
            page_table.set_kv_cpu(req_id = req_id , layer = layer_idx, k_tensor = k_clone , v_tensor = v_clone, cpu_kv_manager = cpu_kv_manager)
            print(f"[Prefill] for req {req_id} has complete for Set_KV_CPU for {layer_idx}") if DEBUG_PREFILL and layer_idx == 0 else None 

            page_table.update_layers_at_cpu(req_id)
            print(f"[Prefill] for req {req_id} has sucessfully incremented the layer {page_table.get_layer_at_cpu(req_id)}") if DEBUG_PREFILL and layer_idx == 0 else None 
            
            if not page_table[req_id]["warmup_done"] and page_table.get_layer_at_cpu(req_id) < 5:
                print(f'''{layer_idx} the {req_id} is put inside schedular queue {page_table.get_layer_at_cpu(req_id)} and {page_table[req_id]["warmup_done"]} ''')
                schedular_queue.put(req_id)
                
            if layer_idx == 0:
                print(f"[Prefill] Layer {layer_idx} CPU KV written. slice={k_clone.flatten()[:10]}") if DEBUG_PREFILL else None

            print(f"[Prefill] for req {req_id} has streamed layer: {layer_idx}") if DEBUG_PREFILL and layer_idx == 0 else None

            return super().update(key_states, value_states, layer_idx, cache_kwargs)
    
    cache = CPUStreamingCache()

    print(f"[Prefill] for req {req_id} has started forward pass") if DEBUG_PREFILL else None

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs, past_key_values=cache, use_cache=True)
    # print(f"[prefill] for req {req_id}  {len(page_table[req_id]["layers"])} ")
    print(f"[Prefill] for req {req_id} has completed forward pass") if DEBUG_PREFILL else None

    # logits_cpu = outputs.logits[:, -1, :].detach().to("cpu")

    # print(f"[Prefill] for req {req_id} has started for set_logits_kv_cpu") if DEBUG_PREFILL else None 
    # page_table.set_logits_kv_cpu(req_id,logits_cpu,cpu_kv_manager)
    # print(f"[Prefill] for req {req_id} has started for set_logits_kv_cpu") if DEBUG_PREFILL else None 
    
    # page_table.table[req_id]["decode_can_start"] = True

    torch.cuda.synchronize(prefill_device)

    prefill_end = datetime.now()
    elapsed_ms = (prefill_end - prefill_start).total_seconds() * 1000

    print(
        f"[{prefill_end.strftime('%H:%M:%S.%f')[:-3]}] Prefill END {req_id} "
        f"(took {elapsed_ms:.2f} ms)"
    ) if DEBUG_PREFILL else None 
    
    