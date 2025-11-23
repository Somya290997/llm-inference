import torch

class GlobalKVCache:
    def __init__(self):
        # key: prompt_hash  → val: {layer_id → (k_block_ids, v_block_ids)}
        self.cache = {}   


    def store_layer(self, prompt_hash, layer_id, k_block_ids, v_block_ids):

        if prompt_hash not in self.cache:
            self.cache[prompt_hash] = {}
        
        self.cache[prompt_hash][layer_id] = {
            "K": k_block_ids,
            "V": v_block_ids
        }
    

    def has_prompt(self, prompt_hash):
        """Quick check if cache has KV for this prompt."""
        return prompt_hash in self.cache


    def get_cached_layers(self, prompt_hash):
        """Returns layer IDs already cached"""

        if prompt_hash not in self.cache:
            return []
        
        return list(self.cache[prompt_hash].keys())


    def get_layer_blocks(self, prompt_hash, layer_id):
        """Get block_ids for K/V of one layer"""

        entry = self.cache[prompt_hash].get(layer_id)

        if entry is None:
            raise KeyError(f"Layer {layer_id} not cached!")
        
        return entry["K"], entry["V"]


    def get_missing_layers(self, prompt_hash, total_layers):
        """Find which layers we still need to compute"""
        
        cached = self.get_cached_layers(prompt_hash)
        return list(set(range(total_layers)) - set(cached))
    


# for layer_id in range(num_layers):
#     # slice per-prompt KV from big batch
#     k_tensor_slice = full_kv[layer_id][:, :, start:end, :]
#     v_tensor_slice = full_kv[layer_id][:, :, start:end, :]

#     # convert to CPU block IDs using cpu_kv_manager
#     k_block_ids, v_block_ids = cpu_kv_manager.write_layer(k_tensor_slice, v_tensor_slice)
#     global_kv_cache.save_kv(prompt_hash, layer_id, k_block_ids, v_block_ids)  # 🚀
    


# prompt_hash = hash(prompt)

# if global_kv_cache.has_prompt(prompt_hash):
#     cached_layers = global_kv_cache.get_layer_ids(prompt_hash)
#     missing = global_kv_cache.get_missing_layers(prompt_hash, total_layers)

#     print(f"[KV Reuse] Hit! Reuse {len(cached_layers)} layers. "
#           f"Only compute {len(missing)} layers.")
    
#     # ⬇️ important: skip computing these layers!
#     skip_layers = set(cached_layers)

# else:
#     skip_layers = set()  # full prefill