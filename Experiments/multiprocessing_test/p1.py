# p1.py
import time
from datetime import datetime

def prefills_come_here(request: dict) -> dict:
    req_id = request["req_id"]
    prompt = request["prompt"]

    print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] P1 START req_id={req_id} prompt='{prompt}'")
    time.sleep(2)  # simulate work time (prefill stage)
    print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] P1 END   req_id={req_id}")

    processed = {
        "req_id": req_id,
        "prefill_output": f"P1 processed: {prompt}"
    }
    return processed