import torch
import time

class BatchedActiveRequest:
    def __init__(self, batch_id, request_ids, past_kv, logits, seq_dtype=torch.float16):
        """
        past_kv: list of (K,V) tensors, each (B, H, T, D)
        logits:  (B, vocab)
        """
        self.batch_id = batch_id
        self.past_kv = past_kv
        self.seq_dtype = seq_dtype
        self.request_ids = request_ids

        B = logits.shape[0]         # total requests in batch
        self.batch_size = B

        self.generated_ids = None   # (B, T)
        self.total_generated = torch.zeros(B, dtype=torch.int32)
        self.finished_mask = torch.zeros(B, dtype=torch.bool)

        # Timing
        self.decode_start_time = time.time()
        self.first_token_time = torch.zeros(B, dtype=torch.float32)    # per-request
        self.tbt_times = [[] for _ in range(B)]  # per-request TBT list


    @staticmethod
    def sample_top_p(logits, temperature=0.8, top_p=0.9):
        logits = logits / temperature
        probs = torch.softmax(logits, dim=-1)

        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        mask = cumulative > top_p
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = False

        probs[mask.scatter(1, sorted_idx, mask)] = 0.0
        probs = probs / probs.sum(dim=-1, keepdim=True)

        return torch.multinomial(probs, num_samples=1)    # (B,1)


    def init_first_token(self, logits, req_start_times):
        next_token = BatchedActiveRequest.sample_top_p(logits)  # (B,1)
        self.generated_ids = next_token.clone()

        now = time.time()
        self.first_token_time = torch.full((self.batch_size,), now, dtype=torch.float32)

        # Compute TTFT per request
        for i in range(self.batch_size):
            ttft_ms = (self.first_token_time[i] - req_start_times[i]) * 1000
            print(f"[TTFT][req {self.batch_id} | idx {i}] = {ttft_ms:.2f} ms")
            

    def update_after_step(self, new_tokens, new_past_kv, eos_token_id):
    
        if self.generated_ids is None:
            self.generated_ids = new_tokens.clone()              # GPU
        else:
            self.generated_ids = torch.cat([self.generated_ids, new_tokens], dim=1)  # GPU
    
        # --- UPDATE COUNTERS ---
        self.total_generated += 1
        self.past_kv = new_past_kv
    
        # --- FIX DEVICE MISMATCH 🚨 ---
        eos_hits = (new_tokens.squeeze(-1) == eos_token_id)      # GPU tensor
        self.finished_mask = self.finished_mask.to(eos_hits.device)   # MOVE ROI request status to GPU
        self.finished_mask |= eos_hits                          # NOW SAFE
    
        # ---- FINISH WHEN ALL TRUE ----
        return self.finished_mask.all().item()