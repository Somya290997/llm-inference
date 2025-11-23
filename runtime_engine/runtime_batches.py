import queue
import threading
import heapq

from datetime import datetime
import time

from page_table.page_table_batches import PageTable
from cpu_kv_manager.cpu_kv_blockmanger_batches import CPUKVBlockManager
from batcher.prefill_batcher import prefill_batches_stage
from prefill_worker.prefill_worker_batches import prefill_stage

# from scheduler.scheduler_2 import scheduler_stage
from transfer_kv.transfer_kv_worker import transfer_stage
from decode_worker.decode_worker_3 import decode_stage
from transfer_kv.transfer_kv_cpu_worker import transfer_kv_cpu_worker

class Runtime:

    def __init__(self):

        # Queue
        self.user_input_queue = queue.Queue()
        self.prefill_queue = queue.Queue()
        
        self.scheduler_queue = []
        
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
        threading.Thread(target=self.KV_write_CPU_worker, args=(self.push_req,) , daemon=True).start()
        threading.Thread(target=self.scheduler_worker, args=() , daemon=True).start()
        threading.Thread(target=self.transfer_worker, args=() , daemon=True).start()
        threading.Thread(target=self.decode_worker, args=() , daemon=True).start()

    # To submit the request
    def submit_request(self, prompt):
        self.user_input_queue.put(prompt)

    # Workers
        
    # user_input_worker
    def user_input_worker(self):
        while True:
            req_id , input_ids , request_prompt = prefill_batches_stage(self.user_input_queue,self.page_table)
            self.prefill_queue.put((req_id , input_ids , request_prompt))
            


    # prefill_worker
    def prefill_worker(self):
        while True:
            req_id , input_ids , request_prompt = self.prefill_queue.get()

            prefill_stage(req_id=req_id, input_ids=input_ids, request_prompt=request_prompt,page_table=self.page_table,cpu_kv_manager=self.cpu_kv_manager, kv_write_to_cpu_queue=self.kv_write_to_cpu_queue)
            

    def KV_write_CPU_worker(self , push_req):
        while True:
            request_for_transfer = self.kv_write_to_cpu_queue.get()
            transfer_kv_cpu_worker(request_for_transfer,self.page_table,self.cpu_kv_manager,self.scheduler_queue, push_req)

    def push_req(self,req_id):
        seq_len = self.page_table[req_id]["seq_len"]
        prefill_done = self.page_table[req_id]["prefill_end_time"]
        age = time.time() - self.page_table[req_id]["req_id_start_time"]
    
        priority = 0
        priority += 100 if prefill_done != 0.0 else 0  # HIGH priority if prefill done
        priority += (1 / seq_len)               # short prompts boost
        priority += (age * 0.01)                # starvation prevention

        heapq.heappush(self.scheduler_queue, (-priority, req_id)) 


    # scheduler_worker
    def scheduler_worker(self):
        while True:
            
            if not self.scheduler_queue:     
                time.sleep(0.001)           
                continue
                
            _, req_id = heapq.heappop(self.scheduler_queue)
    
            action = scheduler_stage(req_id, self.page_table, self.cpu_kv_manager)
    
            if action == "START_TRANSFER":
                self.transfer_queue.put(req_id)
    
            elif action == "WAIT":
                # Wait a bit → then requeue with UPDATED priority
                time.sleep(0.002)
                self.push_req(req_id)
    
            elif action == "RETRY":
                time.sleep(0.005)
                self.push_req(req_id)

            
    # transfer_worker
    def transfer_worker(self):
        while True:
            req_id  = self.transfer_queue.get()
            print(f"[Transfer] Dynamic full transfer for req {req_id}")
            KV_cache , logits = transfer_stage(req_id, self.page_table, self.cpu_kv_manager)
            self.decode_queue.put((req_id, KV_cache ,logits))
    
    # decode worker
    def decode_worker(self):
        while True:
            req_id  , KV_cache , logits = self.decode_queue.get()
            # print( f''' prefill_arrival_rate_ms = inside decode { self.page_table[req_id]["prefill_arrival_rate_ms"] }  and  transfer_rate_ms = {self.page_table[req_id]["transfer_rate_ms"]} is ''')
            decode_stage(req_id,KV_cache,logits,self.page_table,self.cpu_kv_manager )
            


    

        