import torch
import cupy as cp
from transformers.cache_utils import DynamicCache
from datetime import datetime
import time
import yaml

DEBUG_PREFILL = False

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

def prefill_stage(req_id,prompt,page_table,cpu_kv_manager, kv_write_to_cpu_queue):

    torch.cuda.set_device(1)
    print(f"[Prefill] for req {req_id} has been started ")

    enable_p2p()

    print(f"[Prefill] for req {req_id} has Enable P2P ") if DEBUG_PREFILL else None

    model , tokenizer = _load_model_once()

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs_gpu = {k: v.to(prefill_device) for k,v in inputs.items()}
    seq_len = inputs["input_ids"].shape[1]
    
    print(f"Number of tokens in {req_id} is {seq_len} ")
    
    hidden_dim = model.config.hidden_size // model.config.num_attention_heads
    batch_size = 1
    kv_shape = (batch_size, 8, seq_len, hidden_dim)

    print(f"[Prefill] for req {req_id} has started the Page_table_intialization ") if DEBUG_PREFILL else None
    page_table.init_request(req_id = req_id, num_layers=model.config.num_hidden_layers, seq_len=seq_len, shape = kv_shape, dtype=torch.float16)
    print(f"[Prefill] for req {req_id} has complete the Page_table_intialization  ") if DEBUG_PREFILL else None

    page_table[req_id]["req_id_start_time"] = time.time()
    page_table[req_id]["prefill_start_time"] = time.time()
    print(f"Prefill START {req_id}") if DEBUG_PREFILL else None


    class CPUStreamingCache(DynamicCache):
        def update(self, key_states, value_states, layer_idx, cache_kwargs=None):

            # t1 = datetime.now()
            
            # if "hf_kv" not in page_table[req_id]:
            #     page_table[req_id]["hf_kv"] = {}

            k_clone = key_states.detach().contiguous()
            v_clone = value_states.detach().contiguous()
    
            # # Store reference copy (on CPU, already contiguous)
            # page_table[req_id]["hf_kv"][layer_idx] = {
            #     "k": k_clone,  # Make a copy for reference
            #     "v": v_clone,
            # }

            
            # print(f"[Prefill] for req {req_id} shape {key_states.shape}") if DEBUG_PREFILL and layer_idx == 0 else None 
            # print(f"[Prefill] for req {req_id} has started for Set_KV_CPU for {layer_idx}") if DEBUG_PREFILL and layer_idx == 0 else None 
            # page_table.set_kv_cpu(req_id = req_id , layer = layer_idx, k_tensor = k_clone , v_tensor = v_clone, cpu_kv_manager = cpu_kv_manager)
            
            kv_write_to_cpu_queue.put((req_id,layer_idx,k_clone,v_clone))
        
            # print(f"[Prefill] for req {req_id} has complete for Set_KV_CPU for {layer_idx}") if DEBUG_PREFILL and layer_idx == 0 else None 

            # page_table.update_layers_at_cpu(req_id)
            
            print(f"[Prefill] for req {req_id} has sucessfully incremented the layer {page_table.get_layer_at_cpu(req_id)}") if DEBUG_PREFILL and layer_idx == 0 else None 
            
            # if not page_table[req_id]["warmup_done"] and page_table.get_layer_at_cpu(req_id) < 5 and not page_table[req_id]["warmup_scheduled"]:
            #     print(f'''{layer_idx} the {req_id} is put inside schedular queue {page_table.get_layer_at_cpu(req_id)} and {page_table[req_id]["warmup_done"]} ''')
            #     schedular_queue.put(req_id)
                
            # if layer_idx == 0:
            # print(f"[Prefill] Layer {layer_idx} for {req_id} is CPU KV written. slice={k_clone.flatten()[:10]}") if DEBUG_PREFILL else None
            
            # print(f"[Prefill] for req {req_id} has streamed layer: {layer_idx}") if DEBUG_PREFILL and layer_idx == 0 else None
            # end = datetime.now()
            # elapsed_ms1 = (end - t1).total_seconds() * 1000
            # print(f"{elapsed_ms1:.2f} ms is the  for time layer")
            return super().update(key_states, value_states, layer_idx, cache_kwargs)

    
    cache = CPUStreamingCache()

    print(f"[Prefill] for req {req_id} has started forward pass") if DEBUG_PREFILL else None
    
    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs_gpu, past_key_values=cache, use_cache=True)

    print(f"[Prefill] for req {req_id} has completed forward pass") if DEBUG_PREFILL else None

    
    print(f"[Prefill] for req {req_id} has started for set_logits_kv_cpu") if DEBUG_PREFILL else None 
    logits_cpu = outputs.logits[:, -1, :].detach().contiguous()
    
    print(f"Shape of the logits {logits_cpu.shape}") if DEBUG_PREFILL else None 
    
    page_table.set_logits_kv_cpu(req_id,logits_cpu,cpu_kv_manager)
    
    print(f"[Prefill] for req {req_id} has started for set_logits_kv_cpu") if DEBUG_PREFILL else None 
    
    torch.cuda.synchronize(prefill_device)

    page_table[req_id]["prefill_end_time"] = time.time()
    
    # elapsed_ms = (page_table[req_id]["prefill_end_time"] - page_table[req_id]["prefill_start_time"]) * 1000

    # print(
    #     f"[{prefill_end.strftime('%H:%M:%S.%f')[:-3]}] Prefill END {req_id} "
    #     f"(took {elapsed_ms:.2f} ms)"
    # )

    # overlap_time = max(0,(time.time() - page_table[req_id]["CPU_transfer"]) * 1000 )
    # overlap_percentage = (overlap_time / (elapsed_ms) ) * 100
    # print(f'''[Transfer] stage for req {req_id} has been completed with an overlap time of {overlap_time:2f} ms and overlap percentage {overlap_percentage:2f} %''') 

    

    