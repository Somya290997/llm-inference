import torch
import time

# WARMUP_THRESHOLD = 2
no_of_layers = 32
_request_already_in_scheduler = set()

DEBUG_transfer_kv_cpu_worker = True
DEBUG_transfer_kv_cpu_worker2 = False

def transfer_kv_cpu_worker(request ,  page_table , cpu_kv_manager, schedular_queue ):

    global _request_already_in_scheduler

    # print(f"[Transfer_KV_CPU_Worker ] Inside it ") if DEBUG_transfer_kv_cpu_worker else None
    
    req_id , layer_idx , k_clone , v_clone = request

    k_tensor = k_clone.clone()
    v_tensor = v_clone.clone()
    
    if page_table[req_id]["CPU_transfer_start_time"] == 0.0:
        page_table[req_id]["CPU_transfer_start_time"] = time.time()

    print(f"[TRANSFER_KV_CPU] for req {req_id} is started writing for layers: {layer_idx} and the shape is {k_tensor.shape} ") if DEBUG_transfer_kv_cpu_worker2 else None
    
    page_table.set_kv_cpu(req_id = req_id , layer = layer_idx, k_tensor = k_tensor , v_tensor = v_tensor, cpu_kv_manager = cpu_kv_manager)
    page_table.update_layers_at_cpu(req_id)

    print(f"[TRANSFER_KV_CPU] for req {req_id} is completed writing for layers: {layer_idx} and the shape is {k_tensor.shape} ") if DEBUG_transfer_kv_cpu_worker2 else None

    layers_on_cpu = page_table.get_layer_at_cpu(req_id)

    if layer_idx == (no_of_layers-1) and page_table[req_id]["CPU_transfer_end_time"] == 0.0:
         page_table[req_id]["CPU_transfer_end_time"] = time.time()

    if req_id not in _request_already_in_scheduler:
        print(f"[TRANSFER_KV_CPU ] Inside it to put {req_id} in scheduler queue ") if DEBUG_transfer_kv_cpu_worker else None
        _request_already_in_scheduler.add(req_id)
        schedular_queue.put(req_id)
    
