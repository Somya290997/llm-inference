# Import Modules

import torch
from torch.multiprocessing import Process, Queue, Event
# from multiprocessing import Event
from datetime import datetime
import time
import uuid
import os

page_table = {}


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
def worker1(intial_requests_queue: Queue, prefill_ready_queue: Queue, stop_event: Event):
    from  kv_cache_allocator import kv_cache_allocator
    while not stop_event.is_set():
        try:
            request = intial_requests_queue.get()
        except Exception:
            continue
        kv_cache_allocator.kv_cache_allocator(request["req_id"], request["prompt"], page_table)
        prefill_ready_queue.put(request)

# Worker 2 to fetch the req from the prefill_queue (where req_id are already allocated the KV_cache)
def worker2(prefill_ready_queue: Queue, decode_ready_queue: Queue, stop_event: Event):
    from prefill_worker import prefill_worker
    while not stop_event.is_set():
        try:
            request = prefill_ready_queue.get()
        except Exception:
            continue
        prefill_completed_req_id = prefill_worker.prefill_stage(request["req_id"], request["prompt"],page_table)
        decode_ready_queue.put(prefill_completed_req_id)

# Worker 3 takes the prefill done req_id, form the KV cache and starts the decoding it.
def worker3(decode_ready_queue: Queue, output_queue: Queue, stop_event: Event):
    from decode_worker import decode_worker
    while not stop_event.is_set():
        try:
            prefill_request = decode_ready_queue.get()
        except Exception:
            continue
        res = decode_worker.decode_phase(prefill_request,page_table)
        output_queue.put(res)


def result_serving(output_queue: Queue, clear_space: Queue, stop_event: Event):
    while not stop_event.is_set():
        try:
            id , output = output_queue.get()
        except Exception:
            continue
        print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] RESULT {id}")

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

    # Queue for incoming request
    intial_requests_queue = Queue()

    #Queue for reqs that are prefill ready
    prefill_ready_queue = Queue()

    decode_ready_queue = Queue()

    # Queue for reqs ready for decode
    output_queue = Queue()

    # Queue to clear the KV_allocated space
    # clear_space = Queue()

    #Events 
    stop_event = Event()

    # start workers
    p1 = Process(target=worker1, args=(intial_requests_queue, prefill_ready_queue, stop_event), daemon=True)
    p2 = Process(target=worker2, args=(prefill_ready_queue, decode_ready_queue, stop_event), daemon=True)
    pres = Process(target=result_serving, args=(decode_ready_queue, output_queue, stop_event), daemon=True)
    # space = Process(target=clear_worker, args=(clear_space,stop_event), daemon=True)

    p1.start()
    p2.start()
    pres.start()
    # space.start()

    str1 = '''If the main process exits abruptly (e.g. because of an incoming signal), Python’s multiprocessing sometimes fails to clean up its children. It’s a known caveat, so if you’re seeing any resource leaks after interrupting the interpreter, it probably means that this has just happened to you.'''

    user_input(intial_requests_queue, str1,1)
    # user_input(intial_requests_queue, "str2",2)
    # user_input(intial_requests_queue, "str3",3)


    time.sleep(10)
    stop_event.set()
    print("Main process exiting.", flush=True)


    # let them process
    # time.sleep(12)
    # stop_event.set()
    # time.sleep(1)
    # print("Main process exiting.")