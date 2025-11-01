from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from transformers.cache_utils import DynamicCache
from datetime import datetime
import time

tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m")
model = AutoModelForCausalLM.from_pretrained("google/gemma-3-270m")

def prefill_stage(req_id,prompt):
    input = tokenizer(prompt, return_tensors="pt")
    # print(f"The shape of input_ids {(input['input_ids'].shape)}")

    # Create a custom cache class that intercepts updates
    print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] Prefill START {req_id}", flush=True)
    class MonitoredCache(DynamicCache):
        
        def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
            
            torch.save({
                "key": key_states.clone().cpu(),
                "value": value_states.clone().cpu()
            }, f"{req_id}_layer_{layer_idx}.pt")

            
            # print(f"Layer {layer_idx}: Captured KV during computation - Key {key_states.shape}, Value {value_states.shape}")
        
            return super().update(key_states, value_states, layer_idx, cache_kwargs)

    past_key_values = MonitoredCache()

    with torch.no_grad():
        outputs = model(
            **input,
            past_key_values=past_key_values,  
            use_cache=True
        )

    torch.save({
        "last_token_logits" :  outputs.logits[:,-1,:].detach().cpu()
        } , f"{req_id}_last_token_logits.pt"
    )

    print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] Prefill END {req_id}", flush=True)
    return req_id


