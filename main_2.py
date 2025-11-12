from prefill_worker.prefill_worker_2 import prefill_stage
from decode_worker.decode_worker_2 import decode_stage
from common.page_table_2 import PageTable
from cpu_kv_manger.cpu_kv_blockmanger import CPUKVBlockManager

# from models.model_loader import get_model, get_tokenizer

# model = get_model("cuda:1")
# tokenizer = get_tokenizer()

page_table = PageTable()
cpu_kv_manager = CPUKVBlockManager()

if __name__ == "__main__":

    req_id = 1
    str1 = '''Can you tell me something about the Machine Learning? '''

    # Prefill
    prefill_stage(req_id, str1, page_table , cpu_kv_manager)
    # print("PAGE TABLE ENTRY:", page_table[req_id])

    # Decode
    decode_stage(req_id, page_table,cpu_kv_manager)