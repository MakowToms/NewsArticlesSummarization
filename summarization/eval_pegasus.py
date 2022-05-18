# code based on huggingface examples
import json
from pathlib import Path
from typing import Optional

import click
import neptune.new as neptune
import numpy as np
from datasets import load_metric
from tqdm import tqdm
from transformers import PegasusForConditionalGeneration, PegasusTokenizer

from config import NEPTUNE_PROJECT, NEPTUNE_API_TOKEN
from summarization.data import load_sample_data
from subjectivity.filter_subjectivity import load_filtered


# init data
# data = load_sample_data()
# texts = [row["text"] for row in data]
# summaries = [row["summary"] for row in data]
# texts, summaries, filtered_texts = load_filtered("data/newsroom/sample-v2_subj_scored_blob.json", 0.4)
# texts, summaries, filtered_texts = load_filtered("data/newsroom/sample-v2_subj_scored.json", 0.002)

# init model
model_name = "google/pegasus-newsroom"
device = "cpu"
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)

# init metric
metric = load_metric("rouge")


# evaluate
def change_size(predictions, dim):
    new = np.zeros([predictions.shape[0], dim])
    new[:, :predictions.shape[1]] = predictions
    return new


def evaluate(texts, summaries):
    batches = [
        tokenizer(texts[i:(i + 4)], truncation=True, padding="longest", return_tensors="pt").to(device)
        for i in range(0, len(texts), 4)
    ]

    all_preds = []
    for batch in tqdm(batches):
        all_preds.append(model.generate(**batch))

    max_dim = max([preds.shape[1] for preds in all_preds])

    all_preds = [change_size(preds, max_dim) for preds in all_preds]
    preds_concat = np.concatenate(all_preds)
    predicted = tokenizer.batch_decode(preds_concat, skip_special_tokens=True)

    result = metric.compute(predictions=predicted, references=summaries, use_stemmer=True)
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds_concat]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}

    return result


# save results
# result = evaluate(texts, summaries)
# with open(f"data/newsroom/results/sample-v2.json", "w") as f:
#     json.dump(result, f, indent=4)


# result = evaluate(filtered_texts, summaries)
# with open(f"data/newsroom/results/sample-v2_subj_scored_blob-0.4.json", "w") as f:
#     json.dump(result, f, indent=4)


@click.command(help="""
Script to generate Pegasus summaries.

Example usage:
python -m summarization.eval_pegasus -i data/newsroom/sample-v2_subj_scored_blob.json -t 0.4
""")
@click.option(
    "-s",
    "--save-file",
    type=Optional[Path],
    help="Path to save file with results.",
)
@click.option(
    "-i",
    "--input-json",
    type=Path,
    help="Input json with filtered texts.",
)
@click.option(
    "-t",
    "--threshold",
    type=float,
    help="Threshold to filter texts.",
)
def main(
    input_json: Path,
    save_file: Optional[Path] = None,
    threshold: float = 0.01,
):
    params = locals()
    texts, summaries, filtered_texts = load_filtered(input_json, threshold)

    result = evaluate(filtered_texts, summaries)
    if save_file is not None:
        with open(save_file, "w") as f:
            json.dump(result, f, indent=4)

    logger = neptune.init(project=NEPTUNE_PROJECT, api_token=NEPTUNE_API_TOKEN)
    logger["parameters"] = params
    logger["results"] = result
    logger["mean_filtered_length"] = sum([len(t) for t in filtered_texts]) / len(filtered_texts)
    logger["mean_text_length"] = sum([len(t) for t in texts]) / len(filtered_texts)
    logger.stop()


if __name__ == '__main__':
    main()


# device = "cuda" if torch.cuda.is_available() else "cpu"
# src_text = [
#     """ PG&E stated it scheduled the blackouts in response to forecasts for high winds amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow."""
# ]
# assert (
#     tgt_text[0]
#     == "California's largest electricity provider has turned off power to hundreds of thousands of customers."
# )

# def preprocess_function(examples):
#     inputs = [doc for doc in examples["text"]]
#     model_inputs = tokenizer(inputs, max_length=1024, truncation=True)
#     with tokenizer.as_target_tokenizer():
#         labels = tokenizer(examples["summary"], max_length=128, truncation=True)
#     model_inputs["labels"] = labels["input_ids"]
#     return model_inputs
#
#
# tokenized_newsroom = newsroom.map(preprocess_function, batched=True)
