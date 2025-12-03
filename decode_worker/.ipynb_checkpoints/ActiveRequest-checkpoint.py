import time
import torch


class ActiveRequest:
    def __init__(self, unique_id , past_kv):
        self.req_id = unique_id[0]
        self.i = unique_id[1]
        self.past_kv = past_kv   # from prefill
        self.generated_ids = None
        self.finished = False

        self.decode_start_time = None
        self.first_token_time = None
        self.tbt_times = []  # store TBT values
        self.total_generated = 0
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
        
        return torch.multinomial(probs, num_samples=1)
    
    def init_generated(self, page_table, logits):  # first token only
        
        self.decode_start_time = time.time()
        next_token = ActiveRequest.sample_top_p(logits)  # [B,1]
        self.generated_ids = next_token.clone()
        self.first_token_time = time.time()

        ttft_ms = (self.first_token_time -  page_table[self.req_id]["req_id_start_time"]) * 1000
        print(f"[TTFT][req {self.req_id}] and {self.i} {ttft_ms:.2f} ms")

    