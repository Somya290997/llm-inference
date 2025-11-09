# Import Modules

import torch
# from torch.multiprocessing import Process, Queue, Event
# from multiprocessing import Event
import threading
from datetime import datetime
import time
import uuid
import os
from multiprocessing import Manager
import cupy as cp

# Multithreading things.
import threading
from queue import Queue
from threading import Thread, Event

# To queue in the request from the user
def user_input(intial_requests_queue: Queue, line: str, id: int):
    # req_id = str(uuid.uuid4())[:8]
    req_id = id
    intial_requests_queue.put({
        "req_id": req_id,
        "prompt": line
    })
    print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] INPUT added req_id={req_id}")

# Worker1 to fetch the req from intial_request_queus and alocate KV memory
def worker1(page_table: dict, intial_requests_queue: Queue, prefill_ready_queue: Queue, stop_event: Event):

    # To run things in GPU 0
    torch.cuda.set_device(0)

    from  kv_cache_allocator import kv_cache_allocator
    import queue

    while not stop_event.is_set():

        try:
            curr_request = intial_requests_queue.get()
        except queue.Empty:
            continue
        except Exception as e:
            print(f"[Worker1 ERROR] {e}")
            continue

        req_id = curr_request["req_id"]
        request_output = kv_cache_allocator.kv_cache_allocator(req_id, curr_request["prompt"], page_table)
        timeout = 1.0
        start = time.time()
        while req_id not in page_table:
            if time.time() - start > timeout:
                print(f"[Worker1 ERROR] req_id {req_id} still not in page_table after 1s")
                return
            time.sleep(0.01)
            
        print("[Allocator] page_table keys:", page_table.keys())
        prefill_ready_queue.put(request_output)

# Worker 2 to fetch the req from the prefill_queue (where req_id are already allocated the KV_cache)
# def worker2(page_table, prefill_ready_queue: Queue, decode_ready_queue: Queue, stop_event: Event):
#     torch.cuda.set_device(1)
#     from prefill_worker import prefill_worker
#     while not stop_event.is_set():
#         try:
#             request = prefill_ready_queue.get()
#         except Exception:
#             continue
#         prefill_completed_req_id = prefill_worker.prefill_stage(request["req_id"], request["prompt"],page_table)
#         decode_ready_queue.put(prefill_completed_req_id)

def worker2(page_table: dict, prefill_ready_queue: Queue, decode_ready_queue: Queue, stop_event: Event):

    torch.cuda.set_device(1)

    from prefill_worker import prefill_worker
    import queue

    while not stop_event.is_set():
        try:
            curr_request = prefill_ready_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        except Exception as e:
            print(f"[Worker2 ERROR] {e}")
            continue
        
        max_wait = 1.0
        start = time.time()

        while curr_request["req_id"] not in page_table:
            if time.time() - start > max_wait:
                print(f"[Worker2 ERROR] req_id {curr_request['req_id']} still missing in page_table after 1s")
                return
            time.sleep(0.01)
        print(f"{curr_request['req_id']} found in page_table in worker2 ")
        
        prefill_completed_req_id = prefill_worker.prefill_stage(
            curr_request["req_id"], 
            curr_request["prompt"],
            page_table,  
        )

        decode_ready_queue.put(prefill_completed_req_id)


# Worker 3 takes the prefill done req_id, form the KV cache and starts the decoding it.
# def worker3(page_table, decode_ready_queue: Queue, output_queue: Queue, stop_event: Event, decode_model):
#     torch.cuda.set_device(0)
#     from decode_worker import decode_worker
#     while not stop_event.is_set():
#         try:
#             prefill_request = decode_ready_queue.get()
#         except Exception:
#             continue
#         res = decode_worker.decode_phase(prefill_request,page_table)
#         output_queue.put(res)

def worker3(page_table: dict , decode_ready_queue: Queue, output_queue: Queue, stop_event: Event):

    torch.cuda.set_device(0)

    from decode_worker import decode_worker
    import queue

    while not stop_event.is_set():
        try:
            prefill_request = decode_ready_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        except Exception as e:
            print(f"[Worker3 ERROR] {e}")
            continue
        print("Decode Started")
        res = decode_worker.decode_phase(prefill_request['req_id'], page_table)
        print("Decode Ended")
        output_queue.put(res)
        
def result_serving(output_queue: Queue, clear_space: Queue, stop_event: Event):
    import queue
    while not stop_event.is_set():
        try:
            id , output = output_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        except Exception as e:
            print(f"[Result Server ERROR] {e}")
            continue

        with open(f"{id}.txt", 'w') as f:
            f.write(output)

        clear_space.put(id)

# def clear_worker(clear_space: Queue, stop_event: Event):
#     while not stop_event.is_set():
#         try:
#             id = clear_space.get()
#             for i in range(0,18):
#                 os.remove(f"/Users/likhit/Desktop/CU/Fall_25/System_ML/LLM_inference_project/llm-inference/Experiments/streaming_test/{id}_layer_{i}.pt")
            
#             os.remove(f"/Users/likhit/Desktop/CU/Fall_25/System_ML/LLM_inference_project/llm-inference/Experiments/streaming_test/{id}_last_token_logits.pt")

#         except Exception:
#             continue




if __name__ == "__main__":

    page_table = {} 

    import cupy as cp

    for i in range(2):
        for j in range(2):
            if i != j:
                print(f"GPU{i} → GPU{j}: {cp.cuda.runtime.deviceCanAccessPeer(i,j)}")

    # To enable peer to peer processses.

    # for i in range(torch.cuda.device_count()):
    #     for j in range(torch.cuda.device_count()):
    #         if i != j:
    #             can_access = cp.cuda.runtime.deviceCanAccessPeer(i, j)
    #             print(f"[P2P] Can GPU{i} access GPU{j}? {'Yes' if can_access else 'No'}")
    
    #             if can_access:
    #                 try:
    #                     # Switch to device i
    #                     with cp.cuda.Device(i):
    #                         cp.cuda.runtime.deviceEnablePeerAccess(j)
    #                         print(f"[P2P] Enabled GPU{i} → GPU{j}")
    #                 except cp.cuda.runtime.CUDARuntimeError as e:
    #                     # Ignore "peer access already enabled" errors
    #                     if e.status != cp.cuda.runtime.cudaErrorPeerAccessAlreadyEnabled:
    #                         raise

    for i in range(2):
        for j in range(2):
            if i != j:
                try:
                    with cp.cuda.Device(i):
                        cp.cuda.runtime.deviceEnablePeerAccess(j)
                    print(f"[P2P] Enabled {i}->{j}")
                except cp.cuda.runtime.CUDARuntimeError as e:
                    if e.status == cp.cuda.runtime.cudaErrorPeerAccessAlreadyEnabled:
                        print(f"[P2P] Already enabled {i}->{j}")
                    else:
                        print(f"[P2P] ERROR enabling {i}->{j}: {e}")
    
    # Queue for incoming request
    intial_requests_queue = Queue()

    #Queue for reqs that are prefill ready
    prefill_ready_queue = Queue()

    #Queue for reqs that are decode ready
    decode_ready_queue = Queue()

    # Queue for reqs ready for output
    output_queue = Queue()

    # Queue to clear the KV_allocated space
    clear_space = Queue()

    #Events
    stop_event = Event()

    # start workers
    p1 = Thread(target=worker1, args=(page_table, intial_requests_queue, prefill_ready_queue, stop_event), daemon=True)
    p2 = Thread(target=worker2, args=(page_table, prefill_ready_queue, decode_ready_queue, stop_event), daemon=False)
    p3 = Thread(target=worker3, args=(page_table, decode_ready_queue, output_queue, stop_event), daemon=False)

    pres = Thread(target=result_serving, args=(output_queue, clear_space, stop_event), daemon=True)
    # space = Process(target=clear_worker, args=(clear_space,stop_event), daemon=True)

    p1.start()
    p2.start()
    p3.start()
    pres.start()
    # space.start()

    str1 = '''Can you tell me something about the Machine Learning? '''

    user_input(intial_requests_queue, str1,1)
    # user_input(intial_requests_queue, "str2",2)
    # user_input(intial_requests_queue, "str3",3)


    time.sleep(600)
    stop_event.set()
    print("Main process exiting.", flush=True)


    # let them process
    # time.sleep(12)
    # stop_event.set()
    # time.sleep(1)
    # print("Main process exiting.")