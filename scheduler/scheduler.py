import time
import torch
import math
from transfer_kv.transfer_kv_worker import transfer_stage
from transformers.cache_utils import DynamicCache

DEBUG_SCHEDULER = True

def scheduler_stage(req_id, device, dtype, page_table, cpu_kv_manager):

    total_layers = page_table[req_id]["num_layers"]
    WARMUP_THRESHOLD = max(2, total_layers // 4)

    while True:

        layers_on_cpu = page_table.get_layers_at_cpu(req_id)
        prefill_rate = page_table[req_id]["prefill_arrival_rate_ms"]
        transfer_rate = page_table[req_id]["transfer_rate_ms"]

        if layers_on_cpu < WARMUP_THRESHOLD:
            print(f"[Scheduler] has not warm-up transfer for req {req_id}") if DEBUG_SCHEDULER else None
            time.sleep(0.001)
            continue

        if transfer_rate is None:
            # Ask Runtime to start transfer_for_first_few_layers
            print(f"[Scheduler] warm-up transfer rate for req {req_id} is not present yet") if DEBUG_SCHEDULER else None
            return True    # Start warm-up transfer
        
        remaining_prefill = (total_layers - layers_on_cpu) * prefill_rate

        # Full transfer time estimate
        full_transfer_time = total_layers * transfer_rate

        # Ideal delay
        delay = remaining_prefill - full_transfer_time

        if delay > 0:
            print(f"[Scheduler] Delay transfer by {delay:.2f} ms for req {req_id}") if DEBUG_SCHEDULER else None
            time.sleep(delay / 1000)

        print(f"[Scheduler] Dynamic-mode transfer start for req {req_id}")  if DEBUG_SCHEDULER else None
        return True


            
    