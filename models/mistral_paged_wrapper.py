import torch
import torch.nn as nn

class MistralPagedWrapper(nn.Module):
    def __init__(self, model, page_table):
        super().__init__()
        self.model = model
        self.page_table = page_table

    def forward(self, input_ids, req_id):
        hidden = self.model.embed_tokens(input_ids)

        for layer_id, layer in enumerate(self.model.layers):
            hidden = layer(hidden, req_id=req_id, layer_id=layer_id)

        logits = self.model.lm_head(hidden)
        return logits