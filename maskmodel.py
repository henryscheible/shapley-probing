import json
import pickle
from collections import defaultdict

import numpy as np
import torch
from torch import nn


# Copied with some modifications from https://gist.github.com/Helw150/9e9f5320fd49646ac893eec34f41bf0d

def construct_array():
    return []


def bernstein(sample):
    if len(sample) < 2:
        return -1, 1
    mean = np.mean(sample)
    variance = np.std(sample)
    delta = 0.1
    R = 1
    bern_bound = (variance * np.sqrt((2 * np.log(3 / delta))) / len(sample)) + (
        (3 * R * np.log(3 / delta)) / len(sample)
    )
    return mean - bern_bound, mean + bern_bound


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
        self.tracker = open("tracker.txt", "a")

    def track(self, head, acc):
        if head is not None:
            self.contribs[head].append(self.prev - acc)
        else:
            self.baseline = acc
        self.prev = acc
        if self.counter % 100 == 0:
            self.tracker.write(str(self.u.sum()) + "-" + str(self.counter) + "\n")
            self.tracker.flush()
        self.counter += 1

    def finish(self):
        self.tracker.write("Contribution Arrays")
        try:
            pickle.dump(self.contribs)
            self.tracker.write(json.dumps(self.contribs))
        except TypeError:
            print("UNABLE TO WRITE CONTRIBUTION ARRAYS")
        finally:
            self.tracker.close()

    def set_mask(self, mask):
        mask = mask.reshape(12, 12)
        self.head_mask = mask

    def get_head(self, mask):
        head = (mask.reshape(-1) != self.prev_mask.reshape(-1)).nonzero(as_tuple=True)[
            0
        ]
        self.prev_mask = mask
        return head

    def active(self, head):
        def active_memo(head):
            contribs = np.array(self.contribs[head])
            lower, upper = bernstein(contribs)
            if lower > -0.01:
                return False
            elif len(contribs) > self.sample_limit:
                return False
            return True

        stored = self.u[head]
        if head == None:
            return True
        elif stored == 1:
            return False
        else:
            is_active = active_memo(head)
            if is_active:
                return True
            else:
                self.u[head] = 1
                return False

    def reset(self):
        print("RESET")
        self.true_prev = True
        self.prev_mask = torch.ones_like(self.prev_mask).flatten()
        self.head_mask = torch.ones_like(self.head_mask)
        self.prev = self.baseline

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
