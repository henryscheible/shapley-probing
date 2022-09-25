import json
import os
import signal
import sys

import torch
import transformers
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer
from captum.attr import ShapleyValueSampling
import crows_pairs
from maskmodel import MaskModel

from transformers.trainer_callback import PrinterCallback

CHECKPOINT = "henryscheible/crows_pairs_bert"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# Copied with some modifications from https://gist.github.com/Helw150/9e9f5320fd49646ac893eec34f41bf0d

def attribute_factory(model, trainer):
    def attribute(mask):
        mask = mask.flatten()
        model.set_mask(mask)
        acc = trainer.evaluate()["eval_accuracy"]
        return acc

    return attribute


def get_crows_pairs_shapley():
    fake_model = None
    transformers.logging.set_verbosity_error()

    mask = torch.ones((1, 144)).to("cuda")
    model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT)
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    fake_model = MaskModel(model, mask)
    raw_dataset = crows_pairs.load_crows_pairs()
    tokenized_datasets = crows_pairs.process_dataset(raw_dataset, tokenizer)
    eval_dataset = tokenized_datasets["eval"]
    args = TrainingArguments("shapley", log_level="error", disable_tqdm=True)
    trainer = Trainer(
        model=fake_model,
        args=args,
        eval_dataset=eval_dataset,
        compute_metrics=crows_pairs.compute_metrics,
        tokenizer=tokenizer
    )

    trainer.remove_callback(PrinterCallback)

    attribute = attribute_factory(fake_model, trainer)

    with torch.no_grad():
        model.eval()
        sv = ShapleyValueSampling(attribute)
        attribution = sv.attribute(
            torch.ones((1, 144)).to("cuda"), n_samples=1, show_progress=True
        )

    print(attribution)

    with open("contribs.txt", "a") as file:
        file.write(json.dumps(attribution.flatten().tolist()))


if __name__ == "__main__":
    get_crows_pairs_shapley()
