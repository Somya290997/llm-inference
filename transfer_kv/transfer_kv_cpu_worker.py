import torch

WARMUP_THRESHOLD = 2

def transfer_kv_cpu_worker(request ,  page_table , cpu_kv_manager,schedular_queue):
    
    req_id , layer_idx , k_clone , v_clone ,start_time = request

    k_tensor = k_clone.clone()
    v_tensor = v_clone.clone()
        
    page_table.set_kv_cpu(req_id = req_id , layer = layer_idx, k_tensor = k_tensor , v_tensor = v_tensor, cpu_kv_manager = cpu_kv_manager)
    page_table.update_layers_at_cpu(req_id)

    layers_on_cpu = page_table.get_layer_at_cpu(req_id)

    if layers_on_cpu >= WARMUP_THRESHOLD and not page_table[req_id]["warmup_scheduled"]:
        
        page_table[req_id]["warmup_scheduled"] = True
        schedular_queue.put((req_id , start_time))

    

    # if not page_table[req_id]["warmup_done"] and page_table.get_layer_at_cpu(req_id) < 5 and not page_table[req_id]["warmup_scheduled"]:
    #         print(f'''{layer_idx} the {req_id} is put inside schedular queue {page_table.get_layer_at_cpu(req_id)} and {page_table[req_id]["warmup_done"]}''')
    #         schedular_queue.put(req_id)

