from prefill_worker.prefill_worker import prefill_stage
from decode_worker.decode_worker import decode_stage
from common.page_table_2 import PageTable

# from models.model_loader import get_model, get_tokenizer

# model = get_model("cuda:1")
# tokenizer = get_tokenizer()

page_table = PageTable()

if __name__ == "__main__":
    req_id = 1
    prompt = "Hello world"

    # Prefill
    prefill_stage(req_id, prompt, page_table)

    # Decode
    decode_stage(req_id, page_table)