import cupy as cp
import torch

def wrap_ptr_to_tensor(ptr, shape, dtype, device):
    # compute bytes
    nbytes = torch.tensor([], dtype=dtype).element_size()
    for dim in shape:
        nbytes *= dim

    # map dtype
    cp_dtype = {
        torch.float16: cp.float16,
        torch.float32: cp.float32,
        torch.int32: cp.int32,
    }[dtype]

    # wrap raw pointer (no allocation happens!)
    mem = cp.cuda.UnownedMemory(ptr, nbytes, owner=None)
    memptr = cp.cuda.MemoryPointer(mem, 0)

    cp_tensor = cp.ndarray(
        shape=shape,
        dtype=cp_dtype,
        memptr=memptr
    )

    # convert to torch (zero copy)
    return torch.as_tensor(cp_tensor, device=device)