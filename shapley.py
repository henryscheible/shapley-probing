import signal
import sys

import torch
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer
from captum.attr import ShapleyValueSampling
import crows_pairs
from maskmodel import MaskModel

CHECKPOINT = "./test-trainer/checkpoint-450/"


# Copied with some modifications from https://gist.github.com/Helw150/9e9f5320fd49646ac893eec34f41bf0d

def attribute_factory(model, trainer):
    def attribute(mask):
        mask = mask.flatten()
        if mask.sum() == 1:
            model.reset()
        mask = mask == 0  # invert mask order
        if not mask.sum() == 144:
            head = model.get_head(mask)
        else:
            head = None
        if not model.active(head) or mask.sum() <= 72:
            acc = model.prev
            model.true_prev = False
        else:
            if not model.true_prev:
                mask_copy = mask.clone()
                mask_copy[head] = 1
                model.set_mask(mask_copy)
                model.prev = trainer.evaluate()["eval_overall_accuracy"]
            model.set_mask(mask)
            acc = trainer.evaluate()["eval_overall_accuracy"]
            print(acc)
            model.track(head, acc)
            model.true_prev = True
        acc = -1 * acc
        return acc

    return attribute


def get_crows_pairs_shapley():
    fake_model = None

    def signal_handler(signal, frame):
        print("You pressed Ctrl+C!")
        print(signal)  # Value is 2 for CTRL + C
        print(frame)  # Where your execution of program is at moment - the Line Number
        fake_model.finish()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    mask = torch.ones((1, 144)).to("cuda")
    model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT)
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    fake_model = MaskModel(model, mask)
    raw_dataset = crows_pairs.load_crows_pairs()
    tokenized_datasets = crows_pairs.process_dataset(raw_dataset, tokenizer)
    eval_dataset = tokenized_datasets["eval"]
    args = TrainingArguments(CHECKPOINT + "training_args.bin")
    trainer = Trainer(
        model=fake_model,
        args=args,
        eval_dataset=eval_dataset,
        compute_metrics=crows_pairs.compute_metrics,
        tokenizer=tokenizer
    )

    attribute = attribute_factory(fake_model)

    with torch.no_grad():
        model.eval()
        sv = ShapleyValueSampling(attribute)
        attribution = sv.attribute(
            torch.ones((1, 144)).to("cuda"), n_samples=3000, show_progress=True
        )
    fake_model.finish()
