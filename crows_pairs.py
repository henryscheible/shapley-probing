from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, \
    Trainer
import numpy as np
import evaluate


def load_crows_pairs():
    return load_dataset("crows_pairs")['test']


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


def load_model():
    checkpoint = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2).to('cuda')
    return tokenizer, model


def process_dataset(dataset, tokenizer):
    def tokenize_function(example):
        print(example["label"])
        print(example["label"] == 1)
        if example["label"] == 1:
            return tokenizer(example["sent_more"], example["sent_less"], truncation=True)
        else:
            return tokenizer(example["sent_less"], example["sent_more"], truncation=True)

    num_samples = len(dataset["sent_more"])
    dataset = dataset.remove_columns([
        "stereo_antistereo",
        "bias_type",
        "annotations",
        "anon_writer",
        "anon_annotators",
    ])
    dataset = dataset.add_column("label", np.random.choice(2, num_samples))
    tokenized_dataset = dataset.map(tokenize_function, batched=False)
    # Because of the random mixing of sent_more and sent_less using batched=True
    # does not yield a performance gain
    split_tokenized_dataset = tokenized_dataset.train_test_split(
        test_size=0.2
    )
    eval_test_split = split_tokenized_dataset["test"].train_test_split(
        test_size=0.5
    )
    return DatasetDict({
        "train": split_tokenized_dataset["train"],
        "test": eval_test_split["test"],
        "eval": eval_test_split["train"]
    })


def compute_metrics(eval_preds):
    metric = evaluate.load("accuracy")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def train():
    tokenizer, model = load_model()
    raw_dataset = load_crows_pairs()
    tokenized_datasets = process_dataset(raw_dataset, tokenizer)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    training_args = TrainingArguments(
        "test-trainer",
        evaluation_strategy="steps",
        eval_steps=10,
        save_strategy="steps",
        save_steps=10
    )
    trainer = Trainer(
        model,
        training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["eval"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    trainer.train()

