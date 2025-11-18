def transfer_kv_cpu_worker(request , page_table , cpu_kv_manager):
    req_id , layer_idx , k_clone , v_clone = request
    page_table.set_kv_cpu(req_id = req_id , layer = layer_idx, k_tensor = k_clone , v_tensor = v_clone, cpu_kv_manager = cpu_kv_manager)
    page_table.update_layers_at_cpu(req_id)

