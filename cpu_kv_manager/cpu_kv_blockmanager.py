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
    
    def alloc_logits_for_request(self , no_of_elements):

        # Allocate CPU KV for request, but lazy-layer on demand.
     
        block_id = self.next_block_id
        self.logits_pool[block_id] = torch.empty(self.element_per_block,dtype=torch.float16,device='cpu')
        self.next_block_id += 1
        self.elements_filled_in_block[block_id] = no_of_elements
        return block_id

    def write_layer(self, k_tensor, v_tensor):

      
        # Store CPU KV produced by Prefill.

        k_flatten = k_tensor.contiguous().view(-1)
        v_flatten = v_tensor.contiguous().view(-1)
       
        # k_bytes = k_flatten.numel() * k_flatten.element_size()
        # v_bytes = v_flatten.numel() * v_flatten.element_size()

        # no_of_required_blocks_k = math.ceil(k_bytes / self.block_size)
        # no_of_required_blocks_v = math.ceil(v_bytes / self.block_size)


        k_block_ids = []
        v_block_ids = []


        # for i in range(no_of_required_blocks_k):
            
        #     start = i * self.element_per_block
        #     end = start + self.element_per_block
        #     end = min(end,len(k_flatten))
        #     no_of_elements = end - start
        #     block_id = self.alloc_layer_for_request(no_of_elements)

        #     # print(
        #     # f"[DEBUG] alloc size={self.block_pool[block_id].numel()}, "
        #     # f"copy size={len(k_flatten[start:end])}, "
        #     # f"start={start}, end={end}")
        #     # if start >= len(k_flatten): 
        #     #     break
            
        #     self.block_pool[block_id][:no_of_elements] = k_flatten[start:end]
        #     k_block_ids.append(block_id)
        
        # for i in range(no_of_required_blocks_v):
            
        #     start = i * self.element_per_block
        #     end = start + self.element_per_block
        #     end = min(end,len(v_flatten))
        #     no_of_elements = end - start
        #     block_id = self.alloc_layer_for_request(no_of_elements)

        #     # block_id = self.alloc_layer_for_request()
        #     # if start >= len(v_flatten): 
        #     #     break
        #     self.block_pool[block_id][:no_of_elements]  = v_flatten[start:end]
        #     v_block_ids.append(block_id)

        # # print(f"[DEBUG-WRITE] k_tensor.numel()={k_tensor.numel()}, "
        # #   f"v_tensor.numel()={v_tensor.numel()}, "
        # #   f"k_bytes={k_bytes}, block_size={self.block_size}, "
        # #   f"n_blocks_k={no_of_required_blocks_k}, elems_per_block={self.element_per_block}")

        # return k_block_ids , v_block_ids

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

        logits_bytes = logits.numel() * logits.element_size()
        no_of_required_blocks_logits = math.ceil(logits_bytes / self.block_size)
        logits_flatten = logits.flatten()
        logits_block_ids = []

        for i in range(no_of_required_blocks_logits):

            
            start = i * self.element_per_block
            end = start + self.element_per_block
            end =  min(end , len(logits_flatten))
            no_of_elements = end - start
            block_id = self.alloc_logits_for_request(no_of_elements)
            
            # print(
            # f"[DEBUG] alloc size={self.logits_pool[block_id].numel()}, "
            # f"copy size={len(logits_flatten[start:end])}, "
            # f"start={start}, end={end}")
            if start >= len(logits_flatten):  
                break
            
            self.logits_pool[block_id][:no_of_elements] = logits_flatten[start:end]
            logits_block_ids.append(block_id)

        return logits_block_ids
    
    def read_layer(self, k_block_ids, v_block_ids, shape, device):
    
        total_elems = math.prod(shape)
        
        k_tensor = torch.empty(total_elems, dtype=torch.float16, device=device)
        v_tensor = torch.empty(total_elems, dtype=torch.float16, device=device)
    
        # # ----- reconstruct K -----
        # write_ptr = 0
        # for block_id in k_block_ids:
        #     n = self.elements_filled_in_block[block_id]
        #     k_tensor[write_ptr : write_ptr + n] = self.block_pool[block_id][:n]
        #     write_ptr += n
    
        # # ----- reconstruct V -----
        # write_ptr = 0
        # for block_id in v_block_ids:
        #     n = self.elements_filled_in_block[block_id]
        #     v_tensor[write_ptr : write_ptr + n] = self.block_pool[block_id][:n]
        #     write_ptr += n
    
        # return k_tensor.contiguous().view(shape), v_tensor.contiguous().view(shape)

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

        
    def read_logits(self,logits_block_ids,shape,device):

        num_of_elems = math.prod(shape)
        logits = torch.empty(num_of_elems, dtype=torch.float16, device=device)
        
        offset = 0
        for block_id in logits_block_ids:
            n = self.elements_filled_in_block[block_id]
            logits[offset:offset+n] = self.logits_pool[block_id][:n]
            offset += n

        return logits.view(shape)   


    # def free_layer(self, k_block_ids, v_block_ids):

    #     for block_ids in k_block_ids:
    #         del self.block_pool[block_ids]

    #     for block_ids in v_block_ids:
    #         del self.block_pool[block_ids]

    
