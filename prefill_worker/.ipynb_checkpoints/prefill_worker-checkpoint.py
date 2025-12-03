# # Required Modules
# import transformers
# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch
# from transformers.cache_utils import DynamicCache
# from datetime import datetime
# import time
# from models import model_loader
# import yaml
# import cupy as cp

# # Loading the Yaml files
# with open("config/model_config.yaml", "r") as f:
#     config = yaml.safe_load(f)


# # Loading yaml variable and model caching
# decode_device = "cuda:0"
# prefill_device = "cuda:1"
# batch_size = config["batch_size"]
# n_heads = config["n_heads"]
# n_dim = config["n_dim"]
# num_layers = config["num_layers"]
# max_seq_len = 12
# vocab_tokens = config["vocab_tokens"]
# _model = None
# _tokenizer = None


# def _load_model_once():
#     """Load model ONCE when first called"""
#     global _model, _tokenizer
    
#     if _model is None:
#         from models import model_loader
#         print(f"[Prefill] Loading model on cuda:1...")
#         _model = model_loader.get_model("cuda:1")
#         _tokenizer = model_loader.get_tokenizer()
#         print(f"[Prefill] Model loaded ✓")
    
#     return _model, _tokenizer


# def prefill_stage(req_id, prompt, page_table):

#     torch.cuda.set_device(1)
#     model, tokenizer = _load_model_once()
    
#     # tokenizing the input
#     input = tokenizer( prompt,return_tensors="pt",max_length=120,truncation=True).to(prefill_device)
#     max_seq_len = input["input_ids"].shape[1]

#     print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] Prefill START {req_id}", flush=True)

#     kv_cache = {}
#     page_table[req_id] = {
#         "kv_cache" : kv_cache
#     }

#     # Create a custom cache class that intercepts updates
#     class MonitoredCache(DynamicCache):
        
#         def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
            
#             k_tensor = torch.empty([batch_size,n_heads,max_seq_len,n_dim] , dtype=torch.float16, device=decode_device)
#             v_tensor = torch.empty_like(k_tensor)
#             dst_k_ptr = k_tensor.data_ptr()
#             dst_v_ptr = v_tensor.data_ptr()

#             kv_cache[layer_idx] = {
#                 "K_address" : dst_k_ptr,
#                 "V_address" : dst_v_ptr
#             }


#             cp.cuda.runtime.memcpyPeer(
#                 dst_k_ptr, 0,              # destination address (on cuda:0)
#                 key_states.data_ptr(), 1,              # source address (on cuda:1)
#                 key_states.numel() * key_states.element_size()  # size in bytes
#             )

#             if layer_idx == 0:
#                 mem = cp.cuda.UnownedMemory(dst_k_ptr,  key_states.numel() * key_states.element_size(), owner=None)
#                 memptr = cp.cuda.MemoryPointer(mem, 0)
            
#                 cp_arr = cp.ndarray(
#                     shape=(1, 8, 12, 128),
#                     dtype=cp.float16,
#                     memptr=memptr
#                 )
            
#                 dst_from_ptr = torch.as_tensor(cp_arr, device=decode_device)
            
#                 print("Pointer-read slice:", dst_from_ptr.flatten()[:10])
                
                
#             cp.cuda.runtime.memcpyPeer(
#                 dst_v_ptr,   0,            # destination address (on cuda:0)
#                 value_states.data_ptr(),   1,            # source address (on cuda:1)
#                 value_states.numel() * value_states.element_size()  # size in bytes
#             )

#             print(f" layer {layer_idx} has been copied to the dstinations")
        
#             return super().update(key_states, value_states, layer_idx, cache_kwargs)

#     past_key_values = MonitoredCache()

#     with torch.no_grad():
#         outputs = model(
#             **input,
#             past_key_values=past_key_values,  
#             use_cache=True
#         )

#     logits = torch.empty([batch_size,vocab_tokens],dtype=torch.float16, device=decode_device)
#     dst_logits_ptr = logits.data_ptr()

#     page_table[req_id] = {
#         "last_logit_token" : dst_logits_ptr
#     }

#     src_logits_ptr = outputs.logits[:,-1,:]

#     cp.cuda.runtime.memcpyPeer(
#         dst_logits_ptr, 0,
#         src_logits_ptr.data_ptr(), 1,
#         src_logits_ptr.numel() * src_logits_ptr.element_size()
#     )
    
#     print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] Prefill END {req_id}", flush=True)
#     return {"req_id":req_id , "prompt": prompt }

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from transformers.cache_utils import DynamicCache
from datetime import datetime
import cupy as cp
import yaml
import queue
import threading
from models import model_loader

# -----------------------------------------------------------
# Load config
# -----------------------------------------------------------
with open("config/model_config.yaml", "r") as f:
    config = yaml.safe_load(f)

decode_device = "cuda:0"
prefill_device = "cuda:1"

batch_size = config["batch_size"]
n_heads = config["n_heads"]
n_dim = config["n_dim"]
num_layers = config["num_layers"]
vocab_tokens = config["vocab_tokens"]

_model = None
_tokenizer = None


# -----------------------------------------------------------
# Load model once (GPU1)
# -----------------------------------------------------------
def _load_model_once():
    global _model, _tokenizer
    if _model is None:
        print(f"[Prefill] Loading model on cuda:1...")
        _model = model_loader.get_model(prefill_device)
        _tokenizer = model_loader.get_tokenizer()
    return _model, _tokenizer


# -----------------------------------------------------------
# Enable GPU P2P
# -----------------------------------------------------------
def enable_p2p():
    for i in range(2):
        for j in range(2):
            if i != j:
                try:
                    with cp.cuda.Device(i):
                        cp.cuda.runtime.deviceEnablePeerAccess(j)
                        print(f"[P2P] Enabled {i}->{j}")
                except cp.cuda.runtime.CUDARuntimeError as e:
                    if "already" in str(e).lower():
                        print(f"[P2P] Already enabled {i}->{j}")
                    else:
                        print(f"[P2P] ERROR enabling {i}->{j}: {e}")


# -----------------------------------------------------------
# Allocate KV cache on GPU0
# -----------------------------------------------------------
def allocate_kv_cache(req_id, max_seq_len, page_table):

    torch.cuda.set_device(0)
    kv_cache = {}
    tensor_refs = {}

    for layer in range(num_layers):
        K = torch.empty(
            (batch_size, n_heads, max_seq_len, n_dim),
            dtype=torch.float16,
            device=decode_device,
        )
        V = torch.empty_like(K)

        if layer == 0:
            print("ALLOC PTR:", K.data_ptr())

        tensor_refs[layer] = {"K": K, "V": V}

        kv_cache[layer] = {
            "K_address": K.data_ptr(),
            "V_address": V.data_ptr(),
            "shape": K.shape,
        }

    logits_tensor = torch.empty(
        (batch_size, vocab_tokens), dtype=torch.float16, device=decode_device
    )

    page_table[req_id] = {
        "kv_cache": kv_cache,
        "tensor_refs": tensor_refs,
        "logits_tensor": logits_tensor,
        "logits_ptr": logits_tensor.data_ptr(),
    }

    print(f"[Allocator] KV cache allocated ✔ (seq_len={max_seq_len})")


# -----------------------------------------------------------
# Background worker thread
# -----------------------------------------------------------
class TransferWorker:
    def __init__(self):
        self.q = queue.Queue()
        self.stop_flag = False
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        print("[TransferWorker] Started ✓")

    def _loop(self):
        while not self.stop_flag:
            try:
                job = self.q.get(timeout=0.1)
            except queue.Empty:
                continue

            if job is None:
                break

            self._do_copy(job)
            self.q.task_done()

    def _do_copy(self, job):
        layer = job["layer"]
        req_id = job["req_id"]

        # pointers
        src_k_ptr = job["src_k_ptr"]
        src_v_ptr = job["src_v_ptr"]
        dst_k_ptr = job["dst_k_ptr"]
        dst_v_ptr = job["dst_v_ptr"]

        nbytes_k = job["nbytes_k"]
        nbytes_v = job["nbytes_v"]

        tensor_refs = job["tensor_refs"]

        # ✅ copy K/V from GPU1 → GPU0
        with cp.cuda.Device(0):
            cp.cuda.runtime.memcpyPeer(dst_k_ptr, 0, src_k_ptr, 1, nbytes_k)
            cp.cuda.runtime.memcpyPeer(dst_v_ptr, 0, src_v_ptr, 1, nbytes_v)

        torch.cuda.synchronize(0)

        if layer == 0:
            print(f"\n[Worker] Layer0 copy finished → verifying GPU0 slice")
            slice0 = tensor_refs[layer]["K"].flatten()[:10]
            print("GPU0 slice:", slice0)
            if (slice0 == 0).all():
                print("❌ WARNING: still zeros")
            else:
                print("✅ SUCCESS — KV copied")

    def submit(self, job):
        self.q.put(job)

    def wait(self):
        self.q.join()

    def shutdown(self):
        self.stop_flag = True
        self.q.put(None)
        self.thread.join()


_worker = TransferWorker()


# -----------------------------------------------------------
# Async KV cache
# -----------------------------------------------------------
class AsyncKVCache(DynamicCache):

    def __init__(self, req_id, kv_cache, tensor_refs):
        super().__init__()
        self.req_id = req_id
        self.kv_cache = kv_cache
        self.tensor_refs = tensor_refs
        self.source_buffers = {}     # keep contig buffers alive

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):

        # ✅ BLOCK until K/V computes fully for this layer
        torch.cuda.current_stream(prefill_device).synchronize()

        # ✅ make contiguous
        key_c = key_states.contiguous()
        val_c = value_states.contiguous()

        src_k_ptr = key_c.data_ptr()
        src_v_ptr = val_c.data_ptr()

        dst_k_ptr = self.kv_cache[layer_idx]["K_address"]
        dst_v_ptr = self.kv_cache[layer_idx]["V_address"]

        nbytes_k = key_c.numel() * key_c.element_size()
        nbytes_v = val_c.numel() * val_c.element_size()

        # save contig buffer
        if layer_idx not in self.source_buffers:
            self.source_buffers[layer_idx] = []
        self.source_buffers[layer_idx].append((key_c, val_c))

        if layer_idx == 0:
            print("[update] Using CONTIG slice:")
            print("        ", key_c.flatten()[:10])

        # schedule async transfer
        _worker.submit({
            "req_id": self.req_id,
            "layer": layer_idx,
            "src_k_ptr": src_k_ptr,
            "src_v_ptr": src_v_ptr,
            "dst_k_ptr": dst_k_ptr,
            "dst_v_ptr": dst_v_ptr,
            "nbytes_k": nbytes_k,
            "nbytes_v": nbytes_v,
            "tensor_refs": self.tensor_refs,
        })

        return super().update(key_states, value_states, layer_idx, cache_kwargs)


# -----------------------------------------------------------
# Prefill stage
# -----------------------------------------------------------
def prefill_stage(req_id, prompt, page_table):

    torch.cuda.set_device(1)
    model, tokenizer = _load_model_once()

    enable_p2p()

    input = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=120
    ).to(prefill_device)

    max_seq_len = input["input_ids"].shape[1]

    print(f"\n[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] Prefill START {req_id}")

    allocate_kv_cache(req_id, max_seq_len, page_table)
    kv_cache = page_table[req_id]["kv_cache"]
    tensor_refs = page_table[req_id]["tensor_refs"]

    pkv = AsyncKVCache(req_id, kv_cache, tensor_refs)

    print("[Prefill] Running forward pass...")
    with torch.no_grad():
        outputs = model(**input, past_key_values=pkv, use_cache=True)

    print("[Prefill] Forward done. Waiting for KV copies...")
    _worker.wait()

    # copy logits GPU1→GPU0
    dst_logits_ptr = page_table[req_id]["logits_ptr"]
    src_logits = outputs.logits[:, -1, :]
    nbytes = src_logits.numel() * src_logits.element_size()

    with cp.cuda.Device(0):
        cp.cuda.runtime.memcpyPeer(dst_logits_ptr, 0, src_logits.data_ptr(), 1, nbytes)

    print(f"[Prefill] Copied logits ✔ ({nbytes} bytes)")
    print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] Prefill END {req_id}")

    return {"req_id": req_id, "prompt": prompt}


def shutdown():
    _worker.shutdown()