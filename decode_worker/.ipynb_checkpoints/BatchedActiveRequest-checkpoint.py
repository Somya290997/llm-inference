class BatchedActiveRequest:
    def __init__(self, batch_id, past_kv, logits, batch_size):
        self.batch_id     = batch_id      # ONLY ONE req_id
        self.past_kv      = past_kv       # full (B,H,T,D)
        self.batch_size   = batch_size

        self.generated_ids = None         # (B, T)
        self.total_generated = torch.zeros(batch_size, dtype=torch.int32)
        self.finished_mask  = torch.zeros(batch_size, dtype=torch.bool)

        self.decode_start_time = time.time()
        self.first_token_time  = None     # vector of size B
        self.tbt_times = [[] for _ in range(batch_size)]

        
    # COPY YOUR SAME SAMPLING:
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

    def init_first_token(self, logits, req_start_time):
        next_token = sample_top_p(logits)  # (B,1)
        self.generated_ids = next_token.clone()
    
        now = time.time()
        self.first_token_time = torch.full((self.batch_size,), now)
    
        # WE JUST LOG ONE TTFT — SIMPLE ✔
        ttft_ms = (now - req_start_time) * 1000
        print(f"[TTFT][req {self.batch_id}] = {ttft_ms:.2f} ms")