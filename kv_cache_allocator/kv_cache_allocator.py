# Required Modules
import torch
import yaml

# load yaml files
with open("config/model_config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Now you can access values:
batch_size = config["batch_size"]
n_heads = config["n_heads"]
n_dim = config["n_dim"]
num_layers = config["num_layers"]
max_seq_len = 70
vocab_tokens = config["vocab_tokens"]

decode_device = "cuda:0"
prefill_device = "cuda:1"


def kv_cache_allocator(req_id, prompt, page_table):
    # req_id = str(req_id)
    kv_cache = {} 
    for layer_idx in range(int(num_layers)):
        
        # allocate empty space
        k_tensor = torch.empty([batch_size,n_heads,max_seq_len,n_dim] , dtype=torch.float16, device=decode_device)
        v_tensor = torch.empty_like(k_tensor)

        # print(f"key_states device: {k_tensor.device} and shape {k_tensor.shape}")
        # print(f"value_states device: {v_tensor.device} and shape {v_tensor.shape}")
        # print("size", k_tensor.numel() * k_tensor.element_size())
        # final logits shape check..

        # address ptr
        k_ptr = k_tensor.data_ptr()
        v_ptr = v_tensor.data_ptr()

        kv_cache[layer_idx] = {
            "K_address" : k_ptr,
            "V_address" : v_ptr,
            "dtype" : torch.float16,
            "max_seq_len" : max_seq_len
        }

        # print(f"KV cache of layer {layer_idx} is placed on {decode_device}")
    final_layer_logits = torch.empty([batch_size,vocab_tokens],dtype=torch.float16, device=decode_device)
    

    page_table[req_id] = {
        "kv_cache" : kv_cache,
        "last_logit_layer" : final_layer_logits.data_ptr()
    }

    if req_id in page_table:
        print(f"are there any keys in this in KV cache allocator {page_table[req_id].keys()}")
    else:
        print(f"[Error] req_id {req_id} not found in page_table_entry")

    return {"req_id": req_id, "prompt": prompt}

# def kv_cache_allocator(req_id, prompt, page_table):
#     req_id = str(req_id)
#     kv_cache = {}
#     tensor_refs = {}  # ADD THIS - keep tensor references
    
#     for layer_idx in range(num_layers):
#         # allocate empty space
#         k_tensor = torch.empty([batch_size,n_heads,max_seq_len,n_dim] , dtype=torch.float16, device=decode_device)
#         v_tensor = torch.empty_like(k_tensor)
        
#         # STORE TENSOR REFERENCES (NEW)
#         tensor_refs[layer_idx] = {
#             "K_tensor": k_tensor,
#             "V_tensor": v_tensor
#         }
        
#         # address ptr
#         k_ptr = k_tensor.data_ptr()
#         v_ptr = v_tensor.data_ptr()
#         kv_cache[layer_idx] = {
#             "K_address" : k_ptr,
#             "V_address" : v_ptr,
#             "dtype" : torch.float16,
#             "max_seq_len" : max_seq_len
#         }
#         print(f"KV cache of layer {layer_idx} is placed on {decode_device}")
    
#     # final logits shape check
#     final_layer_logits = torch.empty([batch_size,],dtype=torch.float16, device=decode_device)
    
#     page_table[req_id] = {
#         "kv_cache" : kv_cache,
#         "last_logit_layer" : final_layer_logits.data_ptr(),
#         "tensor_refs": tensor_refs,  # STORE THIS (NEW)
#         "logits_tensor": final_layer_logits  # STORE THIS (NEW)
#     }
#     return {"req_id": req_id, "prompt": prompt}