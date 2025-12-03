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
        self.element_per_block = self.block_size // 2
        self.block_pool = {}  # block_id -> Tensor values
        self.elements_filled_in_block = {} # block_id -> Number of elements
        self.logits_pool = {}
        self.next_block_id = 0

    def alloc_layer_for_request(self, no_of_elements):

        # Allocate CPU KV for request, but lazy-layer on demand.
     
        block_id = self.next_block_id
        self.next_block_id += 1
        
        self.block_pool[block_id] = torch.empty(
            no_of_elements,
            dtype=torch.float16,
            device='cpu',
            pin_memory=True
        )
       
        self.elements_filled_in_block[block_id] = no_of_elements
        return block_id

    def alloc_logits_for_request(self, no_of_elements):

        # Allocate CPU KV for request, but lazy-layer on demand.
     
        block_id = self.next_block_id
        self.next_block_id += 1
        
        self.logits_pool[block_id] = torch.empty(
            no_of_elements,
            dtype=torch.float16,
            device='cpu',
            pin_memory=True
        )
       
        self.elements_filled_in_block[block_id] = no_of_elements
        return block_id
    
    def write_layer(self, k_tensor, v_tensor):
      
        # Store CPU KV produced by Prefill.

        k_flatten = k_tensor.contiguous().view(-1)
        v_flatten = v_tensor.contiguous().view(-1)

        k_block_ids = []
        v_block_ids = []

        offset = 0
        while offset < len(k_flatten):
            chunk_size = min(self.element_per_block, len(k_flatten) - offset)
            block_id = self.alloc_layer_for_request(chunk_size)
            
            # Copy data with explicit .copy_() for safety
            self.block_pool[block_id].copy_(k_flatten[offset:offset + chunk_size])
            k_block_ids.append(block_id)
            offset += chunk_size
        
        # ✅ FIX: Process V tensor in chunks
        offset = 0
        while offset < len(v_flatten):
            chunk_size = min(self.element_per_block, len(v_flatten) - offset)
            block_id = self.alloc_layer_for_request(chunk_size)
            
            self.block_pool[block_id].copy_(v_flatten[offset:offset + chunk_size])
            v_block_ids.append(block_id)
            offset += chunk_size

        return k_block_ids, v_block_ids
    
    def write_logits(self, logits):

        logits_flatten = logits.contiguous().view(-1)
        logits_block_ids = []

        offset = 0
        
        while offset < len(logits_flatten):
            chunk_size = min(self.element_per_block, len(logits_flatten) - offset)
            block_id = self.alloc_logits_for_request(chunk_size)

            self.logits_pool[block_id].copy_(logits_flatten[offset:offset + chunk_size])
            logits_block_ids.append(block_id)
            offset += chunk_size
            

        return logits_block_ids
    
    def read_layer(self, k_block_ids, v_block_ids, shape, device):
    
        total_elems = math.prod(shape)
        
        k_tensor = torch.empty(total_elems, dtype=torch.float16, device=device)
        v_tensor = torch.empty(total_elems, dtype=torch.float16, device=device)
    
        write_ptr = 0
        for block_id in k_block_ids:
            n = self.elements_filled_in_block[block_id]
            # ✅ FIX: Use non_blocking=True for async transfer
            k_tensor[write_ptr:write_ptr + n].copy_(
                self.block_pool[block_id][:n],
                non_blocking=True
            )
            write_ptr += n
        
        # Reconstruct V
        write_ptr = 0
        for block_id in v_block_ids:
            n = self.elements_filled_in_block[block_id]
            v_tensor[write_ptr:write_ptr + n].copy_(
                self.block_pool[block_id][:n],
                non_blocking=True
            )
            write_ptr += n
        
        return k_tensor.view(shape), v_tensor.view(shape)

        
    def read_logits(self, logits_block_ids, shape, device):

        total_elems = math.prod(shape)
        logits_tensor = torch.empty(total_elems, dtype=torch.float16, device=device)
        
        logit_ptr = 0
        for block_id in logits_block_ids:
            n = self.elements_filled_in_block[block_id]
            logits_tensor[logit_ptr:logit_ptr + n].copy_(
                self.logits_pool[block_id][:n],
                non_blocking=True
            )
            logit_ptr += n
            
            
        return logits_tensor.view(shape)   

    
