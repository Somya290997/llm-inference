import threading
import queue
import time


# ----------------------
# Shared structures
# ----------------------

page_table = {}     # shared dictionary (thread-safe for simple ops)
alloc_queue = queue.Queue()
prefill_queue = queue.Queue()
decode_queue = queue.Queue()


# ----------------------
# Worker 1: Allocator
# ----------------------
def worker1_allocator():
    while True:
        try:
            req = alloc_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        req_id = req["req_id"]
        print(f"[W1] Allocating KV cache for {req_id}...")

        # simulate allocation
        page_table[req_id] = {
            "kv": [None] * 4,       # pretend 4-layer KV
            "ready": [False] * 4
        }

        print(f"[W1] Added {req_id} to page_table")
        prefill_queue.put(req_id)


# ----------------------
# Worker 2: Prefill
# ----------------------
def worker2_prefill():
    while True:
        try:
            req_id = prefill_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        print(f"[W2] Prefill start for {req_id}")

        # fill each "KV layer"
        for layer in range(4):
            time.sleep(0.2)  # simulate compute
            page_table[req_id]["kv"][layer] = f"layer_{layer}_data"
            page_table[req_id]["ready"][layer] = True
            print(f"[W2] Filled {req_id} layer {layer}")

        print(f"[W2] Prefill done for {req_id}")
        decode_queue.put(req_id)


# ----------------------
# Worker 3: Decode
# ----------------------
def worker3_decode():
    while True:
        try:
            req_id = decode_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        print(f"[W3] Decode start for {req_id}")

        # ensure all layers ready
        while not all(page_table[req_id]["ready"]):
            time.sleep(0.05)

        # simulate decode using KV
        output = " | ".join(page_table[req_id]["kv"])
        print(f"[W3] Decode output for {req_id}: {output}")

        print(f"[W3] Done for {req_id}")


# ----------------------
# Main
# ----------------------
if __name__ == "__main__":
    # Start 3 worker threads
    t1 = threading.Thread(target=worker1_allocator, daemon=True)
    t2 = threading.Thread(target=worker2_prefill, daemon=True)
    t3 = threading.Thread(target=worker3_decode, daemon=True)

    t1.start()
    t2.start()
    t3.start()

    # Send two demo requests
    alloc_queue.put({"req_id": "req1"})
    time.sleep(0.5)
    alloc_queue.put({"req_id": "req2"})

    # Let everything run for a few seconds
    time.sleep(4)
    print("✅ Pipeline demo complete.")