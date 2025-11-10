class PageTable:

    def __init__(self):
        self.table = {}

    def __getitem__(self, req_id):
        return self.table[req_id]

    def __setitem__(self, req_id, value):
        self.table[req_id] = value

    def __contains__(self, req_id):
        return req_id in self.table

    def init_request(self,req_id,num_layers):
        self.table[req_id]={
            "cpu_kv" : {},
            "gpu_kv" : {},
            "seq_len": 0,
            "num_layers": num_layers,
            "logits_cpu": None,
            "logits_gpu_ptr": None,
            "ready_layers": 0,
            "decode_can_start": False,
        }
    
    def set_cpu_kv(self, req_id, layer, k_cpu, v_cpu):
        self.table[req_id]["cpu_kv"][layer] = {
            "K": k_cpu, 
            "V": v_cpu
        }

    def get_cpu_kv(self, req_id, layer):
        return self.table[req_id]["cpu_kv"][layer]

    def set_gpu_kv(self, req_id, layer, k_gpu, v_gpu):
        self.table[req_id]["gpu_kv"][layer] = {
            "K": k_gpu,
            "V": v_gpu,
        }