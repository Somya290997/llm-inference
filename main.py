# Import Modules

import torch
from torch.multiprocessing import Process, Queue, Event
# from multiprocessing import Event
from datetime import datetime
import time
import uuid
import os
from multiprocessing import Manager
import cupy as cp
import queue

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
def worker1(page_table, intial_requests_queue: Queue, prefill_ready_queue: Queue, stop_event: Event):
    torch.cuda.set_device(0)
    from  kv_cache_allocator import kv_cache_allocator
    import queue
    while not stop_event.is_set():
        try:
            request = intial_requests_queue.get()
        except queue.Empty:
            continue
        except Exception as e:
            print(f"[Worker1 ERROR] {e}")
            continue
        req_id = request["req_id"]
        request_output = kv_cache_allocator.kv_cache_allocator(req_id, request["prompt"], page_table)
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

def worker2(page_table, prefill_ready_queue: Queue, decode_ready_queue: Queue, stop_event: Event):
    torch.cuda.set_device(1)
    from prefill_worker import prefill_worker
    import queue
    while not stop_event.is_set():
        try:
            request = prefill_ready_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        except Exception as e:
            print(f"[Worker2 ERROR] {e}")
            continue
        
        # EXTRACT page_table_entry from page_table (using req_id) 
        
        # try:
        #     page_table_entry = page_table[request["req_id"]]
        # except KeyError:
        #     print(f"[Worker2 ERROR] req_id {request['req_id']} not in page_table")
        #     continue
        max_wait = 1.0
        start = time.time()
        while request["req_id"] not in page_table:
            if time.time() - start > max_wait:
                print(f"[Worker2 ERROR] req_id {request['req_id']} still missing in page_table after 1s")
                return
            time.sleep(0.01)
        print(f"{request['req_id']} found in page_table")
        
        prefill_completed_req_id = prefill_worker.prefill_stage(
            request["req_id"], 
            request["prompt"],
            page_table,  # PASS page_table_entry instead of page_table
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

def worker3(page_table, decode_ready_queue: Queue, output_queue: Queue, stop_event: Event):
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
        
        res = decode_worker.decode_phase(prefill_request, page_table)
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
    from multiprocessing import Manager
    import cupy as cp
    
    manager = Manager()
    page_table = manager.dict()

    from models import model_loader
    prefill_model = model_loader.get_model("cuda:1")
    decode_model = model_loader.get_model("cuda:0")
    prefill_tokenizer = model_loader.get_tokenizer()
    
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)

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

    
    # Queue for incoming request
    intial_requests_queue = Queue()

    #Queue for reqs that are prefill ready
    prefill_ready_queue = Queue()

    decode_ready_queue = Queue()

    # Queue for reqs ready for decode
    output_queue = Queue()

    # Queue to clear the KV_allocated space
    clear_space = Queue()

    #Events 
    stop_event = Event()

    # start workers
    p1 = Process(target=worker1, args=(page_table, intial_requests_queue, prefill_ready_queue, stop_event), daemon=True)
    p2 = Process(target=worker2, args=(page_table, prefill_ready_queue, decode_ready_queue, stop_event), daemon=False)
    p3 = Process(target=worker3, args=(page_table, decode_ready_queue, output_queue, stop_event), daemon=False)
    pres = Process(target=result_serving, args=(output_queue, clear_space, stop_event), daemon=True)
    # space = Process(target=clear_worker, args=(clear_space,stop_event), daemon=True)

    p1.start()
    p2.start()
    p3.start()
    pres.start()
    # space.start()

    str1 = '''If the main process exits abruptly (e.g. because of an incoming signal), Python’s multiprocessing sometimes fails to clean up its children. It’s a known caveat, so if you’re seeing any resource leaks after interrupting the interpreter, it probably means that this has just happened to you.'''

    user_input(intial_requests_queue, str1,1)
    # user_input(intial_requests_queue, "str2",2)
    # user_input(intial_requests_queue, "str3",3)


    time.sleep(60)
    stop_event.set()
    print("Main process exiting.", flush=True)


    # let them process
    # time.sleep(12)
    # stop_event.set()
    # time.sleep(1)
    # print("Main process exiting.")