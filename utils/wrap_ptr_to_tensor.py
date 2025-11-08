import cupy as cp
import torch

def wrap_ptr_to_tensor(ptr, shape, dtype, device):
    # 1. Map PyTorch dtype to CuPy dtype
    cp_dtype = {
        torch.float16: cp.float16,
        torch.float32: cp.float32,
        torch.int32: cp.int32
    }[dtype]

    # 2. Wrap raw memory pointer with CuPy
    cp_tensor = cp.ndarray(
        shape=shape,
        dtype=cp_dtype,
        memptr=cp.cuda.MemoryPointer(cp.cuda.Memory(ptr), 0)
    )

    # 3. Convert to PyTorch tensor on given CUDA device
    return torch.as_tensor(cp_tensor, device=device)