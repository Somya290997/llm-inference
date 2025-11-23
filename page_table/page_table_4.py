import torch
import time

DEBUG_PAGE = False

class PageTable:

    def __init__(self):
        self.table = {}
        self.global_cache = {}  # prompt_hash → { req_id, batch_id, prompt_id, kv_info }

    def __getitem__(self, req_id):
        return self.table[req_id]

    def __setitem__(self, req_id, value):
        self.table[req_id] = value

    def __contains__(self, req_id):
        return req_id in self.table
    

    # Request level page_table
    def init_request(self, req_id, batch_size , max_seq_len, num_layers, shape, dtype):

        self.table[req_id] = {
            
            "batch_size": batch_size,
            "num_layers": num_layers,
            "seq_len_total": max_seq_len, 
            "shape": shape,               # (batch_size, n_heads, seq_len_total, head_dim)
            "dtype": dtype,

            "batch_ids": {},
            "layers": {},   
    

            # TIMING / METRICS
            "req_id_start_time": 0.0,
            "req_id_end_time": 0.0,
            "prefill_start_time": 0.0,
            "prefill_end_time": 0.0,
            "cpu_transfer_start_time": 0.0,
            "cpu_transfer_end_time": 0.0,
            "gpu_transfer_start_time": 0.0,
            "gpu_transfer_end_time": 0.0
        }
    
    # per batch_id page_table
    def init_per_batch_request(self, req_id, batch_size , max_seq_len, num_layers, shape, dtype , batch_id):

        req = self.table[req_id]

        req["batch_ids"][batch_id] = {

            "packed_batched" : False,
            "number_of_packed_prompts": 0,
            "prompt_ids" : {}, 
        }


    # per prompt lvl in page_table
    def init_per_prompt_request(self, req_id, batch_size , max_seq_len, num_layers, shape, dtype , batch_ids, prompt_id , token_start_idx , token_end_idx ,prompt  ):

        if req_id not in self.table:
            raise KeyError(f"Request {req_id} not initialized!")

        if batch_ids not in self.table[req_id]["batch_ids"]:
            self.init_per_batch_request(req_id, batch_size , max_seq_len, num_layers, shape, dtype , batch_ids)

        batch_info = self.table[req_id]["batch_ids"][batch_ids]
        
        batch_info["prompt_ids"][prompt_id] = {
            "token_start_idx": token_start_idx,   
            "token_end_idx": token_end_idx, 
            "prompt" : prompt,
            "seq_len": token_end_idx - token_start_idx,

            "prompt_hash": hash(prompt),    # quick compare
            "matched_prefix_req": None,          # reuse from which req?
            "matched_prefix_batch": None,        # reuse from which batch?
            "matched_prefix_prompt": None,       # reuse from which prompt?
            "prefix_len": 0,
                                                      # matched prefix length (tokens)
            "attention_mask": None,

            "last_token_logits_block": None
        }

        if len(batch_info["prompt_ids"]) > 1:
            batch_info["packed_batched"] = True

        batch_info["number_of_packed_prompts"] = len(batch_info["prompt_ids"])

    # ===================
        
    # Function 1: To set the KV from GPU (Prefill) to CPU
        
    def set_kv_on_cpu_layer_wise(self, req_id, num_layer, k_tensor, v_tensor, cpu_kv_manager):

        k_block_ids , v_block_ids = cpu_kv_manager.write_layer(k_tensor,v_tensor)

        self.table[req_id]["layers"][num_layer] = {
            "K": k_block_ids, 
            "V": v_block_ids,
            "numel" : k_tensor.numel()
        }

        total_layers_now = len(self.table[req_id]["layers"])
        print(f"[PageTable] req {req_id} now has {total_layers_now} layers stored.") if DEBUG_PAGE else None
        

    # Function 2: To Copy the KV from CPU to GPU (Decode)
        
    def get_kv_on_gpu_layer_wise(self, req_id, layer,shape, device,cpu_kv_manager):

        while layer not in self.table[req_id]["layers"]:
            time.sleep(0.0002)

        k_block_ids = self.table[req_id]["layers"][layer]["K"]
        v_block_ids = self.table[req_id]["layers"][layer]["V"]

        k_tensor , v_tensor = cpu_kv_manager.read_layer(k_block_ids,v_block_ids,shape,device)
        return k_tensor , v_tensor
    

    # Function 3: To set the last_layer_logits from GPU (Prefill) to CPU

    def set_logits_per_prompt(self, req_id, batch_id, prompt_id, logits, cpu_kv_manager):
        block_id = cpu_kv_manager.write_logits(logits)  # shape: [1, vocab]
        self.table[req_id]["batch_ids"][batch_id]["prompt_ids"][prompt_id]["last_token_logits_block"] = block_id


    # Function 4: To set the last_layer_logits from CPU to GPU (Decode)
        
    def get_logits_per_prompt(self, req_id, batch_id, prompt_id, shape, device, cpu_kv_manager):
        block_id = self.table[req_id]["batch_ids"][batch_id]["prompt_ids"][prompt_id]["last_token_logits_block"]
        logits = cpu_kv_manager.read_logits([block_id], shape, device)
        return logits





















































    # Funtions to update layers at CPU arivals
    def update_layers_at_cpu(self, req_id):
        self.table[req_id]["layers_at_cpu"] += 1
        self.table[req_id]["cpu_layer_timestamps"].append(time.time())
        self.update_prefill_rate(req_id)
        
    def update_prefill_rate(self, req_id):
        ts = self.table[req_id]["cpu_layer_timestamps"]
        if len(ts) < 2:
            return

        diffs = [ts[i] - ts[i - 1] for i in range(1, len(ts))]
        avg_ms = (sum(diffs) / len(diffs)) * 1000

        self.table[req_id]["prefill_arrival_rate_ms"] = avg_ms

    def get_layer_at_cpu(self,req_id):
        return self.table[req_id]["layers_at_cpu"]
    
    # ===================

    # Funtions to update layers at transfer to GPU
    def update_layers_at_transfer(self, req_id , layer_time_ms):
        self.table[req_id]["layers_transfered"] += 1
        self.table[req_id]["transfer_layer_timestamps"].append(layer_time_ms)
        self.update_transfer_rate(req_id)

    def update_transfer_rate(self, req_id):
        times = self.table[req_id]["transfer_layer_timestamps"]
        if len(times) < 2:
            return

        # exponential moving average
        prev = self.table[req_id]["transfer_rate_ms"]
        avg = sum(times) / len(times)

        if prev is None:
            new_rate = avg
        else:
            new_rate = prev * 0.7 + avg * 0.3

        self.table[req_id]["transfer_rate_ms"] = new_rate
    

    def get_layer_at_transfer(self,req_id):
        return self.table[req_id]["layers_transfered"]
    
    # ===================  

    def set_kv_cpu(self, req_id, batch_id , layer, k_tensor, v_tensor, cpu_kv_manager):

        k_block_ids , v_block_ids = cpu_kv_manager.write_layer(k_tensor,v_tensor)

        self.table[req_id]["layers"][layer] = {
            "K": k_block_ids, 
            "V": v_block_ids,
            "numel" : k_tensor.numel()
        }
        total_layers_now = len(self.table[req_id]["layers"])
        print(f"[PageTable] req {req_id} now has {total_layers_now} layers stored.") if DEBUG_PAGE else None
    

    def set_logits_kv_cpu(self,req_id,logits,cpu_kv_manager):

        logits_block_ids = cpu_kv_manager.write_logits(logits)
        self.table[req_id]["last_layer_logits"] = logits_block_ids


    def get_kv_gpu(self, req_id, layer,shape, device,cpu_kv_manager):

        while layer not in self.table[req_id]["layers"]:
            time.sleep(0.0002)

        k_block_ids = self.table[req_id]["layers"][layer]["K"]
        v_block_ids = self.table[req_id]["layers"][layer]["V"]

        k_tensor , v_tensor = cpu_kv_manager.read_layer(k_block_ids,v_block_ids,shape,device)
        return k_tensor , v_tensor
    


