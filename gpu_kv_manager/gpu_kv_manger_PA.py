import torch
import math

class GPUKVBlockManager:
    """
    Stores KV blocks directly on GPU0 (decode device).
    Blocks come from CPUKVBlockManager.
    """

    def __init__(self, device="cuda:0", block_size = 16 * 1024):
        self.device = torch.device(device)
        self.block_size = block_size
        self.gpu_block_pool = {}
        self.elements_in_gpu_block = {}        
        self.next_block_id = 0
    
    def alloc_block(self, num_elements: int):
        """
        Allocate a 1D GPU buffer for `num_elements` float16 values.
        """
        block_id = self.next_block_id
        self.next_block_id += 1

        self.block_pool[block_id] = torch.empty(
            num_elements,
            dtype=torch.float16,
            device=self.device,
        )

        self.elements_in_block[block_id] = num_elements
        return block_id


    def copy_block_from_cpu(self, cpu_tensor: torch.Tensor):

        """
        Copy a CPU tensor into a fresh GPU block.
        cpu_tensor is 1D float16 (just like in CPUKVBlockManager.block_pool[bid])
        """
        flat = cpu_tensor.contiguous().view(-1)
        num_elems = flat.numel()

        block_id = self.alloc_block(num_elems)
        self.block_pool[block_id].copy_(flat, non_blocking=True)
        return block_id


    def get_block(self, block_id: int) -> torch.Tensor:
        """
        Return the 1D GPU tensor associated with this block_id.
        """
        return self.block_pool[block_id]

    def read_blocks_as_tensor(self, block_ids, shape):
        """
        Utility: reconstruct a full tensor of `shape` by concatenating
        all blocks in `block_ids` in order.

        This mirrors CPUKVBlockManager.read_layer, but operates on GPU.
        """
        total_elems = math.prod(shape)
        out = torch.empty(total_elems, dtype=torch.float16, device=self.device)
        ptr = 0

        for bid in block_ids:
            src = self.block_pool[bid]
            n = self.elements_in_block[bid]
            out[ptr:ptr + n].copy_(src[:n], non_blocking=True)
            ptr += n

        return out.view(shape)