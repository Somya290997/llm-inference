import queue
import threading
import heapq
import torch

from datetime import datetime
import time

from page_table.page_table_batches_3 import PageTable
from cpu_kv_manager.cpu_kv_blockmanger_batches import CPUKVBlockManager
from batcher.prefill_batcher_2 import prefill_batches_stage
from prefill_worker.prefill_worker_batches import prefill_stage

from scheduler.scheduler_batches import scheduler_stage
from transfer_kv.transfer_kv_worker_batches import transfer_stage
from decode_worker.ActiveRequest_2 import BatchedActiveRequest

from decode_worker.decode_worker_batches_3 import decode_one_token_batched
from transfer_kv.transfer_kv_cpu_worker_batches import transfer_kv_cpu_worker

class Runtime:

    def __init__(self):

        print("Initializing runtime...")

        # 1) PRELOAD MODELS BEFORE STARTING THREADS
        from models import model_loader
        self.prefill_model = model_loader.get_model("cuda:1")
        self.tokenizer = model_loader.get_tokenizer()
        self.decode_model = model_loader.get_model("cuda:0")

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

        # bin batches
        self.active_decode_pools = {}


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
        if self.page_table.global_metrics["runtime_start_serving"] is None:
            self.page_table.global_metrics["runtime_start_serving"] = time.time()
        self.user_input_queue.put(prompt)

    # Workers
        
    # user_input_worker
    def user_input_worker(self):
        while True:
            req_id , input_ids , request_prompt = prefill_batches_stage(self.user_input_queue,self.page_table,batch_size=16,runtime=self)
            self.max_req_id = max(self.max_req_id,req_id)
            # print(f"About to put inside the prefill queue")
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
                time.sleep(0.002)
                self.scheduler_queue.put(req_id)

    # transfer_worker
    def transfer_worker(self):
        while True:
            req_id = self.transfer_queue.get()
            KV_cache, logits = transfer_stage(req_id, self.page_table, self.cpu_kv_manager)
    
            bin_size = self.page_table[req_id]["bin"]
            if bin_size not in self.active_decode_pools:
                self.active_decode_pools[bin_size] = None
    
            B = logits.shape[0]   # batch size
    
            # 🔥 HERE IS THE FIX — keep **all real request IDs**
            request_ids = [(req_id, i) for i in range(B)]    # 👈 CRUCIAL
    
            batched_req = BatchedActiveRequest(
                batch_id=req_id,
                past_kv=KV_cache,   
                logits=logits,        
                request_ids=request_ids,   # 👈 ADD THIS
                seq_dtype=torch.float16
            )
    
            # ===== FIRST TOKEN DECODE (MANDATORY) =====
            req_start_times = [
                self.page_table[real_id]["req_id_start_time"]
                for real_id in request_ids
            ]
            batched_req.init_first_token(logits, req_start_times)
    
            self.active_decode_pools[bin_size] = batched_req
    
            print(f"[Decode-Ready] req {req_id} with batch={B} added to bin {bin_size}")

    
    # decode worker
    def decode_worker(self):
        while True:
            for bin_size, batched_req in list(self.active_decode_pools.items()):
                if batched_req is None:
                    continue
                    
                finished = decode_one_token_batched(batched_req, self.page_table, self)
                
                if finished:
                    del self.active_decode_pools[bin_size]
                    
            time.sleep(0.001)

    def compute_global_metrics(self, total_req_count):
        gm = self.page_table.global_metrics
    
        # mark serving end once
        if gm["request_count"] == total_req_count and gm["runtime_end_serving"] is None:
            gm["runtime_end_serving"] = time.time()
    
        if gm["request_count"] < 1:
            print("No completed requests yet.")
            return
    
        def avg(x): return sum(x) / len(x) if x else 0.0
    
        print("\n===== SYSTEM-WIDE METRICS =====")
        print(f"Requests Completed        : {gm['request_count']}")
        print(f"Requests Completed         : {gm['requested_completed_with_decode']}")
        print(f"Total Tokens Generated    : {gm['generated_tokens_total']}")
    
        print(f"Avg TTFT                  : {avg(gm['ttft_list']):.2f} ms")
        print(f"Avg TBT                   : {avg(gm['tbt_list']):.2f} ms")
        print(f"Avg Prefill Time          : {avg(gm['prefill_ms']):.2f} ms")
        print(f"Avg Decode Time           : {avg(gm['decode_ms']):.2f} ms")
        print(f"Avg Total Latency         : {avg(gm['total_latency_ms']):.2f} ms")
    
        # ===============================================================
        # 𝟏) FULL WALL TIME (threads + first + last) → total system view
        # ===============================================================
        wall_time_total = time.time() - gm["runtime_start_total"]
        TPS_total = gm["generated_tokens_total"] / wall_time_total
        RPS_total = gm["request_count"] / wall_time_total
    
        print("\n---- THROUGHPUT: TOTAL WALL TIME ----")
        print(f"TPS_total                 : {TPS_total:.2f} tok/s")
        print(f"RPS_total                 : {RPS_total:.2f} req/s")
    
        # ==================================================================
        # 𝟐) ACTIVE SERVING WINDOW ONLY (ignore warmup threads & idle time)
        # ==================================================================
        if gm["runtime_start_serving"] and gm["runtime_end_serving"]:
            serving_time = gm["runtime_end_serving"] - gm["runtime_start_serving"]
            TPS_serving = gm["generated_tokens_total"] / serving_time
            RPS_serving = gm["request_count"] / serving_time
            print("\n---- THROUGHPUT: SERVING WINDOW ----")
            print(f"TPS_serving               : {TPS_serving:.2f} tok/s   (industry measured)")
            print(f"RPS_serving               : {RPS_serving:.2f} req/s")
    
        # =========================================
        # 𝟑) DECODE ONLY TIME (actual GPU compute)
        # =========================================
        if gm["decode_times_total"] > 0:
            TPS_decode = gm["generated_tokens_total"] / gm["decode_times_total"]
            print("\n---- THROUGHPUT: ACTIVE DECODE ----")
            print(f"TPS_decode_only           : {TPS_decode:.2f} tok/s")
        else:
            print("\n---- THROUGHPUT: ACTIVE DECODE ----")
            print(f"TPS_decode_only           : N/A")
    
        tokens_per_req = gm["generated_tokens_total"] / gm["request_count"]
        print("\n---- REQUEST STATS ----")
        print(f"Tokens per Request        : {tokens_per_req:.2f}")
        print("=====================================\n")
                
                

            
        