import os
import torch
from transformers.cache_utils import DynamicCache
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime
import time

# for i in range(0,16):
#     os.remove(f"/Users/likhit/Desktop/CU/Fall_25/System_ML/LLM_inference_project/llm-inference/Experiments/streaming_test/layer_{i}.pt")

def decode_phase(req_id):

    print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] decode START {req_id}", flush=True)
    max_token_len = 120
    num_of_attentions_layers = 18


    # collect the kv cache
    def kv_cache_collection(req_id,num_of_attentions_layers):
        past_key_values = DynamicCache()

        for i in range(0,num_of_attentions_layers):
            data = torch.load(f"{req_id}_layer_{i}.pt",weights_only=True)
            key_states = data["key"]
            value_states = data["value"]
            past_key_values.update(key_states,value_states,i,cache_kwargs=None)
        
        return past_key_values

    #  collect the last logit layer
    def last_token_logits_collection(req_id):
        return torch.load(f"{req_id}_last_token_logits.pt",weights_only=True)["last_token_logits"]


    past_key_values = kv_cache_collection(req_id=req_id,num_of_attentions_layers=num_of_attentions_layers)
    last_token_logits = last_token_logits_collection(req_id=req_id)


    # Checking purpose
    # print(past_key_values)
    # k ,v = past_key_values[0]
    # print(k.shape)
    # print(v.shape)
    # print(last_token_logits.shape)

    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m")
    model = AutoModelForCausalLM.from_pretrained("google/gemma-3-270m")

    generated_words = []

    with torch.no_grad():
        for i in range(max_token_len):
            next_token_id = torch.softmax(last_token_logits,dim=-1)
            logits_val , logis_idx = torch.topk(next_token_id,10,dim=-1)
            top_k_probs = logits_val / torch.sum(logits_val,dim=-1)
            probs = torch.multinomial(top_k_probs,1)
            next_token_id = torch.gather(logis_idx, dim=-1, index=probs)
            word = tokenizer.decode(next_token_id[0])
            generated_words.append(word)
            output_next = model(input_ids=next_token_id,past_key_values=past_key_values)
            last_token_logits = output_next.logits[:,-1,:]
            past_key_values = output_next.past_key_values

    sentence = " ".join(generated_words)

    print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] decode END {req_id}", flush=True)

    return req_id , sentence 






        









