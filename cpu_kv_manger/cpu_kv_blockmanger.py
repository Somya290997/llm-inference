import torch
import math

class CPUKVBlockManager:

    """
    Holds CPU-side KV buffers for staging.
    Prefill writes here.
    Decode copies to GPU when ready.
    """

    def __init__(self, block_size=16*1024):

        self.block_size = block_size
        self.block_pool = {}  # block_id -> Tensor values
        self.logits_pool = {}
        self.next_block_id = 0

    def alloc_layer_for_request(self):

        # Allocate CPU KV for request, but lazy-layer on demand.
     
        block_id = self.next_block_id
        self.block_pool[block_id] = torch.empty(self.block_size//2,dtype=torch.float16,device='cpu')
        self.next_block_id += 1
        return block_id
    
    def alloc_logits_for_request(self):

        # Allocate CPU KV for request, but lazy-layer on demand.
     
        block_id = self.next_block_id
        self.logits_pool[block_id] = torch.empty(self.block_size//2,dtype=torch.float16,device='cpu')
        self.next_block_id += 1
        return block_id

    def write_layer(self, k_tensor, v_tensor):

      
        # Store CPU KV produced by Prefill.
       
        k_bytes = k_tensor.numel() * k_tensor.element_size()
        v_bytes = v_tensor.numel() * v_tensor.element_size()

        no_of_required_blocks_k = math.ceil(k_bytes / self.block_size)
        no_of_required_blocks_v = math.ceil(v_bytes / self.block_size)

        k_flatten = k_tensor.flatten()
        v_flatten = v_tensor.flatten()

        k_block_ids = []
        v_block_ids = []

        for i in range(no_of_required_blocks_k):
            block_id = self.alloc_layer_for_request()
            start = i * (self.block_size // 2)
            end = start + (self.block_size // 2)
            end = min(end,len(k_flatten))
            self.block_pool[block_id][:] = k_flatten[start:end]
            k_block_ids.append(block_id)
        
        for i in range(no_of_required_blocks_v):
            block_id = self.alloc_layer_for_request()
            start = i * (self.block_size // 2)
            end = start + (self.block_size // 2)
            end = min(end,len(v_flatten))
            self.block_pool[block_id][:] = v_flatten[start:end]
            v_block_ids.append(block_id)
        

        return k_block_ids , v_block_ids
    
    def write_logits(self, logits):

        logits_bytes = logits.numel() * logits.element_size()
        no_of_required_blocks_logits = math.ceil(logits_bytes / self.block_size)
        logits_flatten = logits.flatten()
        logits_block_ids = []

        for i in range(no_of_required_blocks_logits):

            block_id = self.alloc_logits_for_request()
            start = i * (self.block_size // 2)
            end = start + (self.block_size // 2)
            end =  min(end , len(logits_flatten))
            self.logits_pool[block_id][:] = logits_flatten[start:end]
            logits_block_ids.append(block_id)

        return logits_block_ids
    
    def read_layer(self, k_block_ids, v_block_ids, shape , device):
        
        total_blocks_k = len(k_block_ids)
        k_tensor = torch.empty( total_blocks_k * (self.block_size // 2) , dtype=torch.float16, device = device)

        total_blocks_v = len(v_block_ids)
        v_tensor = torch.empty( total_blocks_v * (self.block_size // 2) , dtype=torch.float16, device = device)

        for i , block_ids in enumerate(k_block_ids):
            start = i * (self.block_size // 2)
            end = start + (self.block_size // 2)
            k_tensor[start:end] = self.block_pool[block_ids]

        for i , block_ids in enumerate(v_block_ids):
            start = i * (self.block_size // 2)
            end = start + (self.block_size // 2)
            v_tensor[start:end] = self.block_pool[block_ids]
        
        return k_tensor.view(shape) , v_tensor.view(shape)
    
    def read_logits(self,logits_block_ids,shape,device):

        total_blocks_logits = len(logits_block_ids)
        logits_tensor = torch.empty( total_blocks_logits * (self.block_size // 2) , dtype=torch.float16, device = device)

        for i , block_ids in enumerate(logits_block_ids):
            start = i * (self.block_size // 2)
            end = start + (self.block_size // 2)
            logits_tensor[start:end] = self.logits_pool[block_ids]
        
        return logits_tensor.view(shape)


    def free_layer(self, k_block_ids, v_block_ids):

        for block_ids in k_block_ids:
            del self.block_pool[block_ids]

        for block_ids in v_block_ids:
            del self.block_pool[block_ids]

        

    
