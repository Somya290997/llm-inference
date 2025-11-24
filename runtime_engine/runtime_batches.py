import queue
import threading
import heapq

from datetime import datetime
import time

from page_table.page_table_batches import PageTable
from cpu_kv_manager.cpu_kv_blockmanger_batches import CPUKVBlockManager
from batcher.prefill_batcher import prefill_batches_stage
from prefill_worker.prefill_worker_batches import prefill_stage

from scheduler.scheduler_batches import scheduler_stage
from transfer_kv.transfer_kv_worker_batches import transfer_stage

from decode_worker.decode_worker_batches import decode_stage
from transfer_kv.transfer_kv_cpu_worker_batches import transfer_kv_cpu_worker

class Runtime:

    def __init__(self):


        print("Initializing runtime...")

        # 1) PRELOAD MODELS BEFORE STARTING THREADS
        from models import model_loader
        self.prefill_model      = model_loader.get_model("cuda:1")
        self.tokenizer  = model_loader.get_tokenizer()
        self.decode_model       = model_loader.get_model("cuda:0")

        print("Models PRELOADED on GPUs (no cold start!) \n")

        # Queue
        self.user_input_queue = queue.Queue()
        self.prefill_queue = queue.Queue()
        
        self.scheduler_queue = queue.Queue()
        self.max_req_id = 0
        
        self.transfer_queue = queue.Queue()
        self.decode_queue = queue.Queue()
        self.kv_write_to_cpu_queue = queue.Queue()

        self.cache = None

        # Class to manage
        self.cpu_kv_manager = CPUKVBlockManager()
        self.page_table = PageTable()

        # Threads
        threading.Thread(target=self.user_input_worker, args=() , daemon=True).start()
        threading.Thread(target=self.prefill_worker, args=() , daemon=True).start()
        threading.Thread(target=self.KV_write_CPU_worker, args=() , daemon=True).start()
        threading.Thread(target=self.scheduler_worker, args=() , daemon=True).start()
        threading.Thread(target=self.transfer_worker, args=() , daemon=True).start()
        threading.Thread(target=self.decode_worker, args=() , daemon=True).start()

    # To submit the request
    def submit_request(self, prompt):
        print(f"Inside the submit_request")
        self.user_input_queue.put(prompt)

    # Workers
        
    # user_input_worker
    def user_input_worker(self):
        while True:
            req_id , input_ids , request_prompt = prefill_batches_stage(self.user_input_queue,self.page_table,runtime=self)
            self.max_req_id = max(self.max_req_id,req_id)
            print(f"About to put inside the prefill queue")
            self.prefill_queue.put((req_id , input_ids , request_prompt))
            


    # prefill_worker
    def prefill_worker(self):
        while True:
            req_id , input_ids , request_prompt = self.prefill_queue.get()
            prefill_stage(req_id=req_id, input_ids=input_ids, request_prompt=request_prompt,page_table=self.page_table,cpu_kv_manager=self.cpu_kv_manager, kv_write_to_cpu_queue=self.kv_write_to_cpu_queue,runtime=self)
            

    def KV_write_CPU_worker(self):
        while True:
            request_for_transfer = self.kv_write_to_cpu_queue.get()
            transfer_kv_cpu_worker(request_for_transfer,self.page_table,self.cpu_kv_manager,self.scheduler_queue)


    # scheduler_worker
    def scheduler_worker(self):
        while True:

            if self.scheduler_queue.empty():     
                time.sleep(0.001)           
                continue
                
            req_id = self.scheduler_queue.get()
    
            action = scheduler_stage(req_id, self.page_table, self.cpu_kv_manager)
    
            if action == "START_TRANSFER":
                self.transfer_queue.put(req_id)
    
            elif action == "WAIT":
                time.sleep(0.002)
                self.scheduler_queue.put(req_id)
    
            elif action == "RETRY":
                time.sleep(0.005)
                self.scheduler_queue.put(req_id)

            
    # transfer_worker
    def transfer_worker(self):
        while True:
            req_id  = self.transfer_queue.get()
            print(f"[Transfer] Dynamic full transfer for req {req_id}")
            KV_cache , logits = transfer_stage(req_id, self.page_table, self.cpu_kv_manager)
            print(f"[Transfer] Dynamic full transfer for req {req_id} has been completed")
            self.decode_queue.put((req_id, KV_cache , logits))
    
    # decode worker
    def decode_worker(self):
        while True:
            req_id  , KV_cache , logits = self.decode_queue.get()
            decode_stage(req_id,KV_cache,logits,self.page_table,self.cpu_kv_manager,self)



    
            


    

        