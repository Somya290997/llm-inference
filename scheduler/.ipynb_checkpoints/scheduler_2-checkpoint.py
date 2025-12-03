import time
import torch
import math
from transfer_kv.transfer_kv_worker import transfer_stage
from transformers.cache_utils import DynamicCache

DEBUG_SCHEDULER = False
no_of_layers = 32 
THRESHOLD = 2

def scheduler_stage(req_id, page_table, cpu_kv_manager):
    
    # Check if the GPU as space
    free, total = torch.cuda.mem_get_info(device="cuda:0")
    MIN_FREE_BYTES = (math.prod(page_table[req_id]["shape"]) * 2) + 101 
    if free < MIN_FREE_BYTES :
        return "RETRY"

    if page_table.get_layer_at_cpu(req_id) >= THRESHOLD:
        return "START_TRANSFER"
    else:
        return "WAIT"
        
    # to check whether the prefill is complete
    # if page_table[req_id]["prefill_end_time"] != 0.0:
    #     return "START_TRANSFER"

    # est_transfer_time_ms = page_table[
    # est_prefill_time_ms = estimate_prefill_remaining(req_id, page_table)

    # if (est_prefill_time_ms * (no_of_layers - page_table.get_layer_at_cpu(req_id)) <= (est_transfer_time_ms * no_of_layers) :
    #      return "START_TRANSFER"
    # else:
    #      return "WAIT"

    # if page_table.get_layer_at_cpu(req_id) >= THRESHOLD:
    #     return "START_TRANSFER"
    # else:
    #     return "WAIT"