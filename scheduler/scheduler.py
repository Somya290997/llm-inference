import time
import torch
import math
from transfer_kv.transfer_kv_worker import transfer_stage
from transformers.cache_utils import DynamicCache

DEBUG_SCHEDULER = False

def scheduler_stage(req_id, device, page_table, cpu_kv_manager):

    print(f"[Scheduler] inside for req {req_id}") if DEBUG_SCHEDULER else None
    total_layers = page_table[req_id]["num_layers"]

    # if not page_table[req_id]["warmup_done"]:
    #     if layers_on_cpu >= WARMUP_THRESHOLD:
    #         print(f"[Scheduler] inside for req {req_id} has returned warmed ") if DEBUG_SCHEDULER else None
    #         return "warmup"   # run once
    #     else:
    #         print(f"[Scheduler] inside for req {req_id} has returned None ") if DEBUG_SCHEDULER else None
    #         return None

        
    if page_table[req_id]["warmup_done"]:
        
        print(f"[Scheduler] inside for req {req_id} has started full transfer") if DEBUG_SCHEDULER else None
        prefill_rate = page_table[req_id]["prefill_arrival_rate_ms"]
        transfer_rate = page_table[req_id]["transfer_rate_ms"]
        layers_on_cpu = page_table.get_layer_at_cpu(req_id)
        
        if transfer_rate is None:
            return None   # wait for warmup to finish

        # Dynamic time alignment (your previous logic)
        remaining = (total_layers - layers_on_cpu) * prefill_rate
        full_time = total_layers * transfer_rate
        delay = remaining - full_time 

        if delay > 0:
            print(f"[Scheduler] Delay full-transfer by {delay:.2f} ms for req {req_id}")
            time.sleep(delay/1000)

        print(f"[Scheduler] Full transfer start for req {req_id}") if DEBUG_SCHEDULER else None
        return "full"
        
    elif page_table[req_id]["warmup_scheduled"]:
        return "warmup"
        
    else: 
        return None 
            

    return None

            
    