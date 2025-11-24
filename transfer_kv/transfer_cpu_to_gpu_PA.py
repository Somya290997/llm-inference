
def transfer_cpu_to_gpu_for_KV(req_id, page_table, cpu_kv_manager, gpu_kv_manager):

    req_entry = page_table[req_id]
    num_layers = req_entry["num_layers"]

    # to ensure that exists
    req_entry["gpu_blocks"] = req_entry.get("gpu_blocks", {})

    for layer_id in range(num_layers):
        cpu_k_block_ids = req_entry["layers"][layer_id]["K"]
        cpu_v_block_ids = req_entry["layers"][layer_id]["V"]

        gpu_k_ids = []
        gpu_v_ids = []

        for bid in cpu_k_block_ids:
            cpu_block = cpu_kv_manager.block_pool[bid]
            gpu_bid = gpu_kv_manager.copy_block_from_cpu(cpu_block)
            gpu_k_ids.append(gpu_bid)

        for bid in cpu_v_block_ids:
            cpu_block = cpu_kv_manager.block_pool[bid]
            gpu_bid = gpu_kv_manager.copy_block_from_cpu(cpu_block)
            gpu_v_ids.append(gpu_bid)

        # store in page table

        req_entry["gpu_blocks"][layer_id] = {"K": gpu_k_ids, "V": gpu_v_ids}

