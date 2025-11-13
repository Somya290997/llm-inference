from prefill_worker.prefill_worker_2 import prefill_stage
from decode_worker.decode_worker_2 import decode_stage
from common.page_table_2 import PageTable
from cpu_kv_manager.cpu_kv_blockmanager import CPUKVBlockManager

# from models.model_loader import get_model, get_tokenizer

# model = get_model("cuda:1")
# tokenizer = get_tokenizer()

page_table = PageTable()
cpu_kv_manager = CPUKVBlockManager()

if __name__ == "__main__":

    req_id1 = 1
    req_id2 = 2
    req_id3 = 3
    
    str1 = '''Can you explain something about Machine Learning ? '''
    str2 = '''Hi! My name is Likhit? WHats your name? '''
    str3 = ''' Can you Summarise this please ? In modern AI systems, especially those using transformer-based architectures, the size of the input prompt plays an important role in determining latency, memory usage, and the quality of generated responses. When engineers talk about a prompt being around one thousand tokens, they are usually referring to a block of text that is several paragraphs long—roughly four to five thousand characters in length. This amount of text may include system instructions, user queries, constraints, or even examples used in few-shot prompting. Understanding what this looks like in practice helps when designing systems, such as translation pipelines, retrieval-augmented generation frameworks, or multi-process inference architectures, where prompt size directly affects throughput and GPU utilization.

When a transformer model receives a large prompt, it must process all tokens through every layer during the prefill stage. This is expensive, especially for decoder-only models used in LLMs. The prefill phase computes attention across all input tokens, meaning that the computational complexity increases quadratically with the prompt length. For example, a 1024-token prompt requires roughly four times the compute of a 512-token prompt for the same model. Because of this, engineers working on real-time applications, such as speech-to-text translation or conversational agents, must optimize prompt size carefully to balance context and latency.

Another aspect of managing a 1024-token prompt is memory. Each token generates key and value tensors for each attention head in every layer, which are stored in the KV cache. If a model has, for example, 32 layers and 32 attention heads per layer, the KV cache for 1024 tokens can easily exceed several hundred megabytes, depending on the hidden dimension. This is why many high-performance inference frameworks, such as vLLM, FlashAttention-based servers, or custom systems using CUDA streams, focus heavily on KV cache compression, sharing, or streaming. Smaller prompt sizes significantly reduce memory footprint and allow serving more simultaneous requests on a single GPU.

Another important point is prompt engineering. Even when models support long context windows, such as 32k or 128k, not all content contributes equally to the final generation quality. Effective prompting often involves rewriting or summarizing information so that the most relevant parts appear earlier in the sequence. For example, in a retrieval-augmented generation system, retrieved passages may be chunked into smaller segments so that the LLM can focus on the highest-ranking ones rather than blindly receiving large blocks of text.

Finally, understanding prompt length matters when fine-tuning as well. During training, especially when using QLoRA or LoRA adapters, batching large prompt sequences increases GPU memory consumption. Many engineers limit training sequence length to 512 or 1024 tokens for efficiency, unless training a model explicitly meant for long-form reasoning. The trade-offs between sequence length, batch size, and memory often define the maximum throughput of the training loop.

Overall, a 1024-token prompt is long enough to include multiple instructions, several examples, and extensive user context, but short enough to be processed efficiently by most mid-sized LLMs. Understanding its structure and impact is an essential part of building scalable, low-latency AI systems.'''

    # Prefill
    # prefill_stage(req_id1, str1, page_table , cpu_kv_manager)
    # prefill_stage(req_id2, str2, page_table , cpu_kv_manager)
    prefill_stage(req_id3, str3, page_table , cpu_kv_manager)
    # print("PAGE TABLE ENTRY:", page_table[req_id])

    # Decode
    # decode_stage(req_id1, page_table,cpu_kv_manager)
    # decode_stage(req_id2, page_table,cpu_kv_manager)
    decode_stage(req_id3, page_table,cpu_kv_manager)
    