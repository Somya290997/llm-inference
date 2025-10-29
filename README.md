# llm-inference
Project of LLM Inference

Responsibilities:

1. main.py => Entry point starts scheduler + prefill + decode processes.

2. Pagetable_manager => 
    a. creates a shared dict and with helper functions to update_status and get_entry  
    b. create a schema for the page table.

3. Scheduler => 
    a. Pulls from request_queue, forms batch, 
    b. Allocates KV cache space (calls into kv_cache_allocator) 
    c. Adds page table entries for each (req_id, layer) 
    d. Sends “page table reference” to prefill.py to compute K/V

4. prefill_worker =>
    a. Loads model GPU1
    b. For each batch 
        a. Uses FlashAttention2 to compute layer-wise KV
        b. Writes K/V to memory location based on page table info
        c. Updates status to READY in page table

5. decode_worker => 
    a. Continuously polls page table
    b. As soon as status == READY for a (req_id, layer), decodes that layer on GPU 0
    c. Aggregates logits to generate next token
    d. Handles requeueing of requests for next-token generation if needed

6. kv_cache_allocator =>
    a. Handles memory allocation of KV caches
    b. Returns pointers to key_ptr, value_ptr for each request/layer

7. model_config => configures model details

8. Models stored