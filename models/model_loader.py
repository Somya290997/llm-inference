# Import Modules
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training
from accelerate import init_empty_weights
import torch
import os
import yaml

torch.backends.cuda.enable_flash_sdp(True)      # FlashAttention kernel
torch.backends.cuda.enable_math_sdp(False)      # Disable fallback
torch.backends.cuda.enable_mem_efficient_sdp(False)

# Loading the Yaml files
with open("config/model_config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Loading yaml variable and model caching
_model_cache = {}
_tokenizer = None
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models/mistral_7b_4bit_local"))


# get model funtions allows to load the model only once
def get_model(device_id):
    if device_id not in _model_cache:

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )


        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            # quantization_config=bnb_config,
            device_map={"": device_id},
            torch_dtype=torch.float16,
            local_files_only=True
        ).eval()

        _model_cache[device_id] = model

        # 💥 Enable FlashAttention (only if supported)
        if hasattr(model.config, "_attn_implementation"):
            model.config._attn_implementation = "flash_attention_2"
            print("FlashAttention is supported by this model")
        else:
            print("[WARN] FlashAttention not supported by this model")
        
    return _model_cache[device_id]

# get tokenizer funtions allows to load the tokenizer only once
def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH,local_files_only=True,use_fast=False)
    return _tokenizer
