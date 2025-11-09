import threading
import torch
import cupy as cp
import time

SRC_DEV = 1
DST_DEV = 0

SHAPE = (1, 8, 128, 128)
DTYPE = torch.float32


def num_bytes(shape, dtype):
    elem = torch.tensor([], dtype=dtype).element_size()
    count = 1
    for s in shape:
        count *= s
    return elem * count


def producer(shared):
    torch.cuda.set_device(SRC_DEV)
    print("[Producer] Creating ZERO tensor on GPU1")

    # ✅ src = all zeros
    src = torch.zeros(SHAPE, dtype=DTYPE, device=f"cuda:{SRC_DEV}")

    print("SRC tensor slice:", src.flatten()[:10])

    shared["src_ptr"] = src.data_ptr()
    shared["nbytes"]  = num_bytes(SHAPE, DTYPE)
    shared["shape"]   = SHAPE

    print("[Producer] ptr:", src.data_ptr())
    print("[Producer] Ready!")

    return 


def consumer(shared):
    torch.cuda.set_device(DST_DEV)
    print("[Consumer] Waiting for producer...")

    needed = ["src_ptr", "nbytes", "shape"]
    while not all(k in shared for k in needed):
        time.sleep(0.01)

    src_ptr = shared["src_ptr"]
    nbytes  = shared["nbytes"]
    shape   = shared["shape"]

    print("[Consumer] Got metadata!")
    print("  ptr:", src_ptr)
    print("  nbytes:", nbytes)
    print("  shape:", shape)

    # ✅ dst starts as RANDOM (so we can see change)
    dst = torch.randn(shape, dtype=DTYPE, device=f"cuda:{DST_DEV}")
    print("DST BEFORE memcpyPeer:", dst.flatten()[:10])

    # ✅ Copy from GPU1 → GPU0
    print("[Consumer] memcpyPeer...")
    cp.cuda.runtime.memcpyPeer(
        dst.data_ptr(), DST_DEV,
        src_ptr,        SRC_DEV,
        nbytes
    )

    torch.cuda.synchronize(DST_DEV)

    print("DST AFTER memcpyPeer:", dst.flatten()[:10])

    # --------------------------------------------------
    # ✅ Pointer-only RECONSTRUCTION using DLPack
    # --------------------------------------------------
    print("[Consumer] Reconstructing from raw pointer...")

    with cp.cuda.Device(DST_DEV):
        mem = cp.cuda.UnownedMemory(dst.data_ptr(), nbytes, owner=None)
        memptr = cp.cuda.MemoryPointer(mem, 0)
        cp_arr = cp.ndarray(shape=shape, dtype=cp.float32, memptr=memptr)

        # ✅ safe zero-copy conversion
        dst_from_ptr = torch.utils.dlpack.from_dlpack(cp_arr.toDlpack())

    print("Pointer-read slice:", dst_from_ptr.flatten()[:10])

    return


if __name__ == "__main__":
    shared = {}

    t_prod = threading.Thread(target=producer, args=(shared,))
    t_cons = threading.Thread(target=consumer, args=(shared,))

    t_prod.start()
    t_cons.start()

    t_prod.join()
    t_cons.join()

    print("✅ Done")