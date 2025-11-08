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
max_seq_len = 120
vocab_tokens = config["vocab_tokens"]

decode_device = "cuda:0"
prefill_device = "cuda:1"

def kv_cache_allocator(req_id, prompt, page_table):
    kv_cache = {}
    for layer_idx in range(num_layers):

        # allocate empty space
        k_tensor = torch.empty([batch_size,n_heads,max_seq_len,n_dim] , dtype=torch.float16, device=decode_device)
        v_tensor = torch.empty_like(k_tensor)
        
        # final logits shape check..
        final_layer_logits = torch.empty([batch_size,],dtype=torch.float16, device=decode_device)

        # address ptr
        k_ptr = k_tensor.data_ptr()
        v_ptr = v_tensor.data_ptr()

        kv_cache[layer_idx] = {
            "K_address" : k_ptr,
            "V_address" : v_ptr,
            "dtype" : torch.float16,
            "max_seq_len" : max_seq_len
        }

        print(f"KV cache of layer {layer_idx} is placed on {decode_device}")


    page_table[req_id] = {
        "kv_cache" : kv_cache,
        "last_logit_layer" : final_layer_logits.data_ptr()
    }    