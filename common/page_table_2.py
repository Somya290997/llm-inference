import torch

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
            "logits" : [],
            "num_layers": num_layers,
            "seq_len" : seq_len,
            "shape": shape,
            "dtype": dtype,
            "allocated": 0,
            "decode_can_start": False,
        }

    def set_kv_cpu(self, req_id, layer, k_tensor, v_tensor, cpu_kv_manager):

        k_block_ids , v_block_ids = cpu_kv_manager.write_layer(k_tensor,v_tensor)

        self.table[req_id]["layers"][layer] = {
            "K": k_block_ids, 
            "V": v_block_ids
        }
    

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
        

    def free_req(self,req_id,cpu_kv_manager):

        for layer_id in range(self.table[req_id]["num_layers"]):
            k_block_ids = self.table[req_id][layer_id]["K"]
            v_block_ids = self.table[req_id][layer_id]["V"]
            cpu_kv_manager.free_layer(k_block_ids,v_block_ids)

        del self.table[req_id]
