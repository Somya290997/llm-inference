import time
import torch
from datetime import datetime

DEBUG_DECODE = False
decode_device = "cuda:0"


# -----------------------------
# -------- SAMPLING -----------
# -----------------------------
def sample_top_p(logits, temperature=0.8, top_p=0.9):
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)

    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    mask = cumulative > top_p
    mask[..., 1:] = mask[..., :-1].clone()
    mask[..., 0] = False

    # zero-out masked elements
    probs = torch.where(
        mask.scatter(1, sorted_idx, mask),
        torch.zeros_like(probs),
        probs
    )
    probs = probs / probs.sum(dim=-1, keepdim=True)

    return torch.multinomial(probs, 1)  # (B,1)



# -----------------------------
# ------ ACTIVE BATCH CLASS ---
# -----------------------------
class BatchedActiveRequest:
    def __init__(self, batch_id, past_kv, logits, batch_size):
        self.batch_id = batch_id     # one req_id per batch
        self.past_kv = past_kv       # list of (B,H,T,D)
        self.batch_size = batch_size

        self.generated_ids = None              # (B, T)
        self.total_generated = torch.zeros(batch_size, dtype=torch.int32)
        self.finished_mask = torch.zeros(batch_size, dtype=torch.bool)

        # timing
        self.decode_start_time = time.time()
        self.first_token_time = None           # vector (B,)
        self.tbt_times = [[] for _ in range(batch_size)]


    def init_first_token(self, logits, req_start_times):
        next_token = sample_top_p(logits)  # (B,1)
        self.generated_ids = next_token.clone()

        now = time.time()
        self.first_token_time = torch.full((self.batch_size,), now , dtype=torch.float64 )
        print(req_start_times)

        # SAME PRINT FORMAT AS YOUR ORIGINAL CODE
        req_start_times = torch.tensor(req_start_times, dtype=torch.float64)  # convert list → tensor
        ttft_ms = (self.first_token_time - req_start_times) * 1000
    
        for i, t in enumerate(ttft_ms):
            print(f"[TTFT][req {self.batch_id} | idx {i}] = {t:.2f} ms")



# -----------------------------
# ------- ONE TOKEN STEP ------
# -----------------------------
def decode_one_token_batched(batched_req, page_table, runtime):
    torch.cuda.set_device(0)
    model = runtime.decode_model
    tokenizer = runtime.tokenizer

    input_ids = batched_req.generated_ids[:, -1:]  # (B,1)
    past_kv   = batched_req.past_kv              # list of (B,H,T,D)

    step_start = time.time()

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            past_key_values=past_kv,
            use_cache=True
        )

    page_table[batched_req.batch_id]["decode_start_time"] = time.time()

    next_logits = outputs.logits[:, -1, :]         # (B,vocab)
    next_tokens = sample_top_p(next_logits)        # (B,1)
    new_past_kvs = outputs.past_key_values

    # 1) Append tokens
    batched_req.generated_ids = torch.cat(
        [batched_req.generated_ids, next_tokens], dim=1
    )

    # 2) Update past_kv
    batched_req.past_kv = new_past_kvs

    # 3) Update counters
    batched_req.total_generated += 1
    tbt_ms = (time.time() - step_start) * 1000

    for i in range(batched_req.batch_size):
        batched_req.tbt_times[i].append(tbt_ms)

    # 4) EOS check
    eos = tokenizer.eos_token_id
    eos_hits = (next_tokens.squeeze(-1) == eos)
    batched_req.finished_mask = batched_req.finished_mask.to(eos_hits.device)
    batched_req.finished_mask |= eos_hits

    early_stop_mask = batched_req.total_generated >= 200
    batched_req.finished_mask |= early_stop_mask.to(batched_req.finished_mask.device)
    # batched_req.finished_mask |= eos_hits

    # if batched_req.total_generated[0] % 50 == 0:   # every 50 tokens
    #     print("[DEBUG] total_generated:", batched_req.total_generated)
    #     print("[DEBUG] finished_mask  :", batched_req.finished_mask)

    # If ALL finished → save metrics like before
    if batched_req.finished_mask.all().item():
        end_time = time.time()
        req = page_table[batched_req.batch_id]  # only ONE req_id

        req["Decode_start_time"] = batched_req.decode_start_time
        req["Decode_end_time"]   = end_time
        req["req_id_end_time"]   = end_time
        req["generated_tokens"]  = int(batched_req.total_generated.sum().item())

        B = batched_req.batch_size 

        ttft_ms   = (batched_req.first_token_time[0] - req["prefill_start_time"]) * 1000
        prefill_ms = (req["prefill_end_time"] - req["prefill_start_time"]) * 1000
        decode_ms  = (end_time - batched_req.decode_start_time) * 1000
        total_ms   = (end_time - req["req_id_start_time"]) * 1000

        gm = page_table.global_metrics

        batch_tbt_avgs = []
        for i in range(B):
            if len(batched_req.tbt_times[i]) > 0:
                avg_tbt = sum(batched_req.tbt_times[i]) / len(batched_req.tbt_times[i])
                batch_tbt_avgs.append(avg_tbt)
    
        if batch_tbt_avgs:
            gm["tbt_list"].append(sum(batch_tbt_avgs) / len(batch_tbt_avgs))
        else:
            gm["tbt_list"].append(0.0)

        print(ttft_ms, " : " , prefill_ms , " : " , decode_ms)

        # global storage — SAME AS YOUR LOGIC 
        gm["ttft_list"].append(ttft_ms)
        gm["prefill_ms"].append(prefill_ms)
        gm["decode_ms"].append(decode_ms)
        gm["total_latency_ms"].append(total_ms)
        gm["generated_tokens_total"] += req["generated_tokens"]
        gm["decode_times_total"]    += (end_time - batched_req.decode_start_time)
        gm["requested_completed_with_decode"] += batched_req.batch_size  
        # after metrics are stored...
        gm["request_count"] += batched_req.batch_size  
        

        print(f"[Batch-Finished] req {batched_req.batch_id} — done")
        return True

    return False