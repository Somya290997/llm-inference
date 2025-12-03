import torch.nn as nn

class MistralPagedWrapper(nn.Module):
    def __init__(self, full_model, page_table):
        super().__init__()
        self.full_model = full_model        # has lm_head ✔
        self.model = full_model.model       # only layers & embeddings
        self.page_table = page_table

    def forward(self, input_ids, req_id):
        hidden = self.model.embed_tokens(input_ids)

        for layer_id, layer in enumerate(self.model.layers):
            residual = hidden
            hidden = layer.input_layernorm(hidden)

            hidden, _, _ = layer.self_attn(
                hidden,
                req_id=req_id,
                layer_id=layer_id
            )
            hidden = hidden + residual

            residual = hidden
            hidden = layer.post_attention_layernorm(hidden)
            hidden = layer.mlp(hidden)
            hidden = hidden + residual

        # NOW USE full_model.lm_head (works!)
        return self.full_model.lm_head(hidden)