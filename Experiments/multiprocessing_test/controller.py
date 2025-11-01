# main.py
from multiprocessing import Process, Queue, Event
from p1 import prefills_come_here
from p2 import decode_come_here
from datetime import datetime
import time
import uuid


def user_input_p1(requests_queue: Queue, line: str):
    req_id = str(uuid.uuid4())[:8]
    requests_queue.put({
        "req_id": req_id,
        "prompt": line
    })
    print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] INPUT added req_id={req_id}")


def worker1(requests_queue: Queue, prefill_queue: Queue, stop_event: Event):
    while not stop_event.is_set():
        try:
            request = requests_queue.get(timeout=0.5)
        except Exception:
            continue
        prefill_res = prefills_come_here(request)
        prefill_queue.put(prefill_res)


def worker2(prefill_queue: Queue, output_queue: Queue, stop_event: Event):
    while not stop_event.is_set():
        try:
            prefill_request = prefill_queue.get(timeout=0.5)
        except Exception:
            continue
        res = decode_come_here(prefill_request)
        output_queue.put(res)


def result_serving(output_queue: Queue, stop_event: Event):
    while not stop_event.is_set():
        try:
            output = output_queue.get(timeout=0.5)
        except Exception:
            continue
        print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] RESULT {output['req_id']} -> {output['output']}")


if __name__ == "__main__":
    requests_queue = Queue()
    prefill_queue = Queue()
    output_queue = Queue()
    stop_event = Event()

    # start workers
    p1 = Process(target=worker1, args=(requests_queue, prefill_queue, stop_event), daemon=True)
    p2 = Process(target=worker2, args=(prefill_queue, output_queue, stop_event), daemon=True)
    pres = Process(target=result_serving, args=(output_queue, stop_event), daemon=True)

    p1.start()
    p2.start()
    pres.start()

    # simulate 3 requests arriving close together
    for i in range(3):
        user_input_p1(requests_queue, f"prompt #{i+1}")
        time.sleep(1)

    # let them process
    time.sleep(12)
    stop_event.set()
    time.sleep(1)
    print("Main process exiting.")