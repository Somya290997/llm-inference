# p2.py
import time
from datetime import datetime

def decode_come_here(prefill_obj: dict) -> dict:
    req_id = prefill_obj["req_id"]
    prefill_text = prefill_obj["prefill_output"]

    print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] P2 START req_id={req_id}")
    time.sleep(3)  # simulate slower work (decode stage)
    print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] P2 END   req_id={req_id}")

    final_output = {
        "req_id": req_id,
        "output": f"P2 decoded -> {prefill_text}"
    }
    return final_output