import queue
import threading

from common.page_table_2 import PageTable
from cpu_kv_manager.cpu_kv_blockmanager import CPUKVBlockManager

from prefill_worker.prefill_worker_3 import prefill_stage
from scheduler.scheduler import scheduler_stage
from transfer_kv.transfer_kv_worker import transfer_stage
from decode_worker.decode_worker_3 import decode_stage

class Runtime:

    def __init__(self):

        # Queue
        self.user_input_queue = queue.Queue()
        self.prefill_queue = queue.Queue()
        self.schedular_queue = queue.Queue()
        self.transfer_queue = queue.Queue()
        self.decode_queue = queue.Queue()

        # Class to manage
        self.cpu_kv_manager = CPUKVBlockManager()
        self.page_table = PageTable()

        # Threads
        threading.Thread(target=self.user_input_worker, args=() , daemon=True).start()
        threading.Thread(target=self.prefill_worker, args=() , daemon=True).start()
        threading.Thread(target=self.scheduler_worker, args=() , daemon=True).start()
        threading.Thread(target=self.transfer_worker, args=() , daemon=True).start()
        threading.Thread(target=self.decode_worker, args=() , daemon=True).start()

    # To submit the request
    def submit_request(self,req_id , prompt):
        while True:
            self.user_input_queue.put((req_id,prompt))


    # Workers
        
    # user_input_worker
    def user_input_worker(self):
        while True:
            req_id , prompt = self.user_input_queue.get()
            self.prefill_queue.put((req_id,prompt))


    # prefill_worker
    def prefill_worker(self):
        while True:
            req_id , prompt = self.prefill_queue.get()
            self.schedular_queue.put(req_id)
            prefill_stage(req_id=req_id,prompt=prompt,page_table=self.page_table,cpu_kv_manager=self.cpu_kv_manager)


    # scheduler_worker
    def scheduler_worker(self):
        req_id = self.schedular_queue.get()
        device = device # to much

        while True:
            if scheduler_stage(req_id, device, self.page_table, self.cpu_kv_manager):
                self.transfer_queue.put(req_id)


            
    # transfer_worker
    def transfer_worker(self):
        while True:
            req_id = self.transfer_queue.get()
            warm_up_layers = 3
            total_layers = self.page_table[req_id]["num_layers"]

            if self.page_table[req_id]["transfer_rate_ms"] is None:
                print(f"[Transfer] Warm-up transfer for req {req_id}")
                transfer_stage(req_id, 0, warm_up_layers - 1, 
                            self.page_table, self.cpu_kv_manager)
                # After this, transfer_rate_ms now exists
                # Tell scheduler to run again
                self.schedular_queue.put(req_id)
                continue

            print(f"[Transfer] Dynamic full transfer for req {req_id}")
            KV_cache = transfer_stage(req_id, warm_up_layers, total_layers - 1,
                        self.page_table, self.cpu_kv_manager)

            # Mark finished
            self.page_table[req_id]["ready_for_decode"] = True

            # Send req_id to decode worker
            self.decode_queue.put((req_id,KV_cache))
    
    # decode worker
    def decode_worker(self):
        while True:
            req_id  , KV_cache = self.decode_queue.get()
            decode_stage(req_id,KV_cache,self.page_table,self.cpu_kv_manager)


    

        