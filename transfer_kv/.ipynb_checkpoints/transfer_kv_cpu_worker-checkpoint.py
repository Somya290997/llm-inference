import torch
import time

# WARMUP_THRESHOLD = 2
no_of_layers = 32
request_already_in_scheduler = set()

def transfer_kv_cpu_worker(request ,  page_table , cpu_kv_manager, schedular_queue , push_req ):
    
    req_id , layer_idx , k_clone , v_clone = request

    k_tensor = k_clone.clone()
    v_tensor = v_clone.clone()
    
    if page_table[req_id]["CPU_transfer_start_time"] == 0.0:
        page_table[req_id]["CPU_transfer_start_time"] = time.time()
    
    page_table.set_kv_cpu(req_id = req_id , layer = layer_idx, k_tensor = k_tensor , v_tensor = v_tensor, cpu_kv_manager = cpu_kv_manager)
    page_table.update_layers_at_cpu(req_id)

    layers_on_cpu = page_table.get_layer_at_cpu(req_id)

    if layer_idx == (no_of_layers-1) and page_table[req_id]["CPU_transfer_end_time"] == 0.0:
         page_table[req_id]["CPU_transfer_end_time"] = time.time()

    if req_id not in request_already_in_scheduler:
        push_req(req_id)
        request_already_in_scheduler.add(req_id)
    
