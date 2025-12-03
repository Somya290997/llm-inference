from runtime_engine.runtime_batches_PA import Runtime
import time

runtime = Runtime()

prompt1 = '''Tell me something facts about India?'''

runtime.submit_request(prompt1)

while True:
    time.sleep(5)
    