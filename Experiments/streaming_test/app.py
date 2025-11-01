from multiprocessing import Process, Queue, Event
from datetime import datetime
import time
import uuid
from prompt import str1, str2, str3
import os

def user_input(requests_queue: Queue, line: str, id: int):
    # req_id = str(uuid.uuid4())[:8]
    req_id = id
    requests_queue.put({
        "req_id": req_id,
        "prompt": line
    })
    print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] INPUT added req_id={req_id}")


def worker1(requests_queue: Queue, prefill_queue: Queue, stop_event: Event):
    from prefill import prefill_stage
    while not stop_event.is_set():
        try:
            request = requests_queue.get()
        except Exception:
            continue
        prefill_res = prefill_stage(request["req_id"], request["prompt"] )
        prefill_queue.put(prefill_res)


def worker2(prefill_queue: Queue, output_queue: Queue, stop_event: Event):
    from decode import decode_phase
    while not stop_event.is_set():
        try:
            prefill_request = prefill_queue.get()
        except Exception:
            continue
        res = decode_phase(prefill_request)
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

def clear_worker(clear_space: Queue, stop_event: Event):
    while not stop_event.is_set():
        try:
            id = clear_space.get()
            for i in range(0,18):
                os.remove(f"/Users/likhit/Desktop/CU/Fall_25/System_ML/LLM_inference_project/llm-inference/Experiments/streaming_test/{id}_layer_{i}.pt")
            
            os.remove(f"/Users/likhit/Desktop/CU/Fall_25/System_ML/LLM_inference_project/llm-inference/Experiments/streaming_test/{id}_last_token_logits.pt")

        except Exception:
            continue




if __name__ == "__main__":

    requests_queue = Queue()
    prefill_queue = Queue()
    output_queue = Queue()
    clear_space = Queue()
    stop_event = Event()

    # start workers
    p1 = Process(target=worker1, args=(requests_queue, prefill_queue, stop_event), daemon=True)
    p2 = Process(target=worker2, args=(prefill_queue, output_queue, stop_event), daemon=True)
    pres = Process(target=result_serving, args=(output_queue, clear_space, stop_event), daemon=True)
    space = Process(target=clear_worker, args=(clear_space,stop_event), daemon=True)

    p1.start()
    p2.start()
    pres.start()
    space.start()


    user_input(requests_queue, str1,1)
    user_input(requests_queue, str2,2)
    user_input(requests_queue, str3,3)


    time.sleep(50)
    stop_event.set()
    print("Main process exiting.", flush=True)


    # let them process
    # time.sleep(12)
    # stop_event.set()
    # time.sleep(1)
    # print("Main process exiting.")