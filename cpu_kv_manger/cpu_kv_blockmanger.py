import torch

class CPUKVBlockManager:
    """
    Holds CPU-side KV buffers for staging.
    Prefill writes here.
    Decode copies to GPU when ready.
    """

    def __init__(self):
        self.cpu_kv = {}  # req_id → layer → (K,V)

    def alloc_for_request(self, req_id, num_layers, shape, dtype):
        """
        Allocate CPU KV for request, but lazy-layer on demand.
        """
        self.cpu_kv[req_id] = {
            "num_layers": num_layers,
            "shape": shape,
            "dtype": dtype,
            "layers": {}
        }

    def write_layer(self, req_id, layer_idx, k_cpu, v_cpu):
        """
        Store CPU KV produced by Prefill.
        """
        self.cpu_kv[req_id]["layers"][layer_idx] = (k_cpu, v_cpu)


    def get_layer(self, req_id, layer_idx):
        return self.cpu_kv[req_id]["layers"][layer_idx]