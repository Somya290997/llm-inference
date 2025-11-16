import torch
import time


DEBUG_PAGE = False

class PageTable:

    def __init__(self):
        self.table = {}

    def __getitem__(self, req_id):
        return self.table[req_id]

    def __setitem__(self, req_id, value):
        self.table[req_id] = value

    def __contains__(self, req_id):
        return req_id in self.table

    def init_request(self,req_id, num_layers, seq_len, shape, dtype = torch.float16):

        self.table[req_id]={
            "layers" : {},
            "input_ids" : None,
            "num_layers": num_layers,
            "seq_len" : seq_len,
            "shape": shape,
            "dtype": dtype,
            "layers_at_cpu": 0,
            "cpu_layer_timestamp": [],  # prefill to CPU
            "prefill_arrival_rate_ms": None,  # avg ms per layer
            "transfer_layer_timestamps": [],  # CPU → GPU (decode) transfer times
            "transfer_rate_ms": None,         # avg ms per layer
            "transfer_in_progress": False,    # avoid double scheduling
            "layers_transfered" : 0,
            "ready_for_decode": False,        # decode can begin
            "kv_gpu_bytes": 0           # estimated KV memory size for scheduling
        }


    def update_layers_at_cpu(self, req_id):
        self.table[req_id]["layers_at_cpu"] += 1
        self.table[req_id]["cpu_layer_timestamp"].append(time.time())
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
    
    def update_layers_at_transfer(self, req_id , layer_time_ms):
        self.table[req_id]["layers_transfered"] += 1
        self.table[req_id]["transfer_layer_timestamps"].append(layer_time_ms)
        self.update_transfer_rate(req_id)

    
    def update_transfer_rate(self, req_id):
        times = self.table[req_id]["transfer_layer_times_ms"]
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

    def set_kv_cpu(self, req_id, layer, k_tensor, v_tensor, cpu_kv_manager):

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
        self.table[req_id]["logits"] = logits_block_ids


    def get_kv_gpu(self, req_id, layer,shape, device,cpu_kv_manager):

        k_block_ids = self.table[req_id]["layers"][layer]["K"]
        v_block_ids = self.table[req_id]["layers"][layer]["V"]

        k_tensor , v_tensor = cpu_kv_manager.read_layer(k_block_ids,v_block_ids,shape,device)
        return k_tensor , v_tensor
    
    def get_logits_kv_gpu(self,req_id,device,shape,cpu_kv_manager):

        logits_block_ids = self.table[req_id]["logits"]
        logits = cpu_kv_manager.read_logits(logits_block_ids,shape,device)
        return logits
        

    # def free_req(self,req_id,cpu_kv_manager):

    #     for layer_id in range(self.table[req_id]["num_layers"]):
    #         k_block_ids = self.table[req_id][layer_id]["K"]
    #         v_block_ids = self.table[req_id][layer_id]["V"]
    #         cpu_kv_manager.free_layer(k_block_ids,v_block_ids)

    #     del self.table[req_id]
