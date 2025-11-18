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

        self.cache = None

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
        print(f"I am at {req_id}")
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
            self.cache = prefill_stage(req_id=req_id,prompt=prompt,page_table=self.page_table,cpu_kv_manager=self.cpu_kv_manager,schedular_queue=self.schedular_queue)


    # scheduler_worker
    def scheduler_worker(self):
        while True:
            req_id = self.schedular_queue.get()
            device = "cuda:1" # to much
            action = scheduler_stage(req_id, device, self.page_table, self.cpu_kv_manager)
            if action == "warmup":
                self.page_table[req_id]["warmup_done"] = True
                self.transfer_queue.put(("warmup", req_id))
            elif action == "full":
                self.transfer_queue.put(("full", req_id))


            
    # transfer_worker
    def transfer_worker(self):
        while True:
            mode, req_id = self.transfer_queue.get()
            warm_up_layers = 3
            total_layers = self.page_table[req_id]["num_layers"]
            
            if mode == "warmup":
                # print(f"[Transfer] Warm-up for req {req_id} has started")
                transfer_stage(req_id, 0, warm_up_layers,
                               self.page_table, self.cpu_kv_manager)
                # self.page_table[req_id]["warmup_done"] = True
                # print(f"[Transfer] Warm-up for req {req_id} has been done")
                # self.table[req_id]["layers_transfered"] = 0
                # DO NOT push back to scheduler
                # prefill will eventually push again when layers accumulate
                self.schedular_queue.put(req_id)
                continue
    
            if mode == "full":
                print(f"[Transfer] Dynamic full transfer for req {req_id}")
                self.page_table[req_id]["full_transfer_scheduled"] = True
                KV_cache , logits = transfer_stage(req_id, 0, total_layers,
                                          self.page_table, self.cpu_kv_manager)
    
                self.page_table[req_id]["ready_for_decode"] = True
                self.decode_queue.put((req_id, KV_cache ,logits ))
    
    # decode worker
    def decode_worker(self):
        while True:
            req_id  , KV_cache , logits = self.decode_queue.get()
            decode_stage(req_id,KV_cache,logits,self.page_table,self.cpu_kv_manager)


    

        