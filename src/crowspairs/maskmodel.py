import json
import pickle
from collections import defaultdict

import numpy as np
import torch
from torch import nn


# Copied with some modifications from https://gist.github.com/Helw150/9e9f5320fd49646ac893eec34f41bf0d

def construct_array():
    return []

class MaskModel(nn.Module):
    def __init__(self, real_model, head_mask):
        super(MaskModel, self).__init__()
        self.contribs = defaultdict(construct_array)
        self.counter = 0
        self.prev = 1.0
        self.real_model = real_model
        self.head_mask = head_mask
        self.true_prev = True
        self.sample_limit = 1000
        self.prev_mask = torch.ones_like(head_mask).to("cuda").flatten()
        self.u = torch.zeros_like(head_mask).to("cuda").flatten()
        self.tracker = open("out.txt", "a")

    def set_mask(self, mask):
        mask = mask.reshape(12, 12)
        self.head_mask = mask


    # Copied from transformers.models.bert.modeling_bert.BertModel.forward
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        return self.real_model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            head_mask=self.head_mask,
        )
