import time
import torch
import math
from transfer_kv.transfer_kv_worker import transfer_stage
from transformers.cache_utils import DynamicCache

DEBUG_SCHEDULER = False
no_of_layers = 32 
THRESHOLD = 2

def scheduler_stage(req_id, page_table, cpu_kv_manager):

    print(f"[SCHEDULER] inside it for req_id  {req_id}") if DEBUG_SCHEDULER else None
    
    # Check if the GPU as space
    free, total = torch.cuda.mem_get_info(device="cuda:0")
    MIN_FREE_BYTES = (math.prod(page_table[req_id]["shape"]) * 2) + 101 
    
    if free < MIN_FREE_BYTES :
        print(f"No Free Space")
        return "RETRY"

    if page_table.get_layer_at_cpu(req_id) >= THRESHOLD:
        return "START_TRANSFER"
    else:
        return "WAIT"

    return None