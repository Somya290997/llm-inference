from runtime_engine.runtime_batches import Runtime
import time

def calculate_throughput(page_table):
    completed = 0
    total_time = 0.0

    for req_id, entry in page_table.table.items():
        if entry["Decode_start_time"] != 0.0:  # request finished
            start = entry["req_id_start_time"]
            end = entry["Decode_start_time"]
            total_time += (end - start)
            completed += entry["batch_size"]

    if completed == 0:
        return print("No completed requests yet")

    avg_time = total_time / completed
    qps = completed / total_time   # requests / total seconds

    print(f"\n📊 Throughput Report")
    print(f"Total completed requests: {completed}")
    print(f"Avg request time: {avg_time*1000:.2f} ms")
    print(f"🔥 Throughput = {qps:.3f} requests/sec")


def calculate_token_throughput(page_table):
    total_tokens = 0
    total_decode_time = 0.0

    for req_id, entry in page_table.table.items():
        if entry["Decode_end_time"] != 0.0:
            total_tokens += entry["generated_tokens"]
            total_decode_time += (entry["Decode_end_time"] - entry["Decode_start_time"])

    if total_decode_time == 0:
        return print("No decode finished yet.")

    tps = total_tokens / total_decode_time
    print(f"🔥 Token Throughput = {tps:.2f} tokens/sec")
    
def final_performance_report(page_table):
    calculate_throughput(page_table)       # QPS
    calculate_token_throughput(page_table) # tokens/sec

runtime = Runtime()

prompt1 = '''Tell me something facts about India?'''
prompt2 = '''Tell me something facts about Antratica do people live there?'''
prompt3 = '''Tell me something facts about America and is guns common there?'''
prompt4 = '''Tell me something facts about Asia and do people love living there?'''
prompt5 = '''Tell me something facts about Africa?'''

runtime.submit_request(prompt1)
runtime.submit_request(prompt1)
runtime.submit_request(prompt2)
runtime.submit_request(prompt3)
runtime.submit_request(prompt4)
runtime.submit_request(prompt5)
runtime.submit_request(prompt1)
runtime.submit_request(prompt2)
runtime.submit_request(prompt3)
runtime.submit_request(prompt4)
runtime.submit_request(prompt5)

runtime.submit_request(prompt1)
runtime.submit_request(prompt2)
runtime.submit_request(prompt3)
runtime.submit_request(prompt4)
runtime.submit_request(prompt5)

runtime.submit_request(prompt1)
runtime.submit_request(prompt2)
runtime.submit_request(prompt3)
runtime.submit_request(prompt4)
runtime.submit_request(prompt5)

runtime.submit_request(prompt1)
runtime.submit_request(prompt2)
runtime.submit_request(prompt3)
runtime.submit_request(prompt4)
runtime.submit_request(prompt5)

runtime.submit_request(prompt1)
runtime.submit_request(prompt2)
runtime.submit_request(prompt3)
runtime.submit_request(prompt4)
runtime.submit_request(prompt5)

runtime.submit_request(prompt1)
runtime.submit_request(prompt2)
runtime.submit_request(prompt3)
runtime.submit_request(prompt4)
runtime.submit_request(prompt5)


time.sleep(100)

page_table = runtime.page_table
last_req = runtime.max_req_id

while runtime.page_table[last_req]["req_id_end_time"] == 0.0:
    time.sleep(1)
    
print("All requests finished!")
final_performance_report(page_table)

while True:
    time.sleep(10)