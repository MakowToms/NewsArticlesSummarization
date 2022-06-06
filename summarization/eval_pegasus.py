# code based on huggingface examples
import json
from collections import defaultdict
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
from subjectivity.filter_subjectivity import load_filtered, load_random_as_many_as_filtered

# init data
# data = load_sample_data()
# texts = [row["text"] for row in data]
# summaries = [row["summary"] for row in data]
# texts, summaries, filtered_texts = load_filtered("data/newsroom/sample-v2_subj_scored_blob.json", 0.4)
# texts, summaries, filtered_texts = load_filtered("data/newsroom/sample-v2_subj_scored.json", 0.002)

# init metric
metric = load_metric("rouge")


# evaluate
def change_size(predictions, dim):
    new = np.zeros([predictions.shape[0], dim])
    new[:, :predictions.shape[1]] = predictions
    return new


def evaluate(texts, summaries, tokenizer, model, batch_size, device, evaluate_single):
    batches = [
        tokenizer(texts[i:(i + batch_size)], truncation=True, padding="longest", return_tensors="pt").to(device)
        for i in range(0, len(texts), batch_size)
    ]

    all_preds = []
    for batch in tqdm(batches):
        all_preds.append(model.generate(**batch).cpu())

    max_dim = max([preds.shape[1] for preds in all_preds])

    all_preds = [change_size(preds, max_dim) for preds in all_preds]
    preds_concat = np.concatenate(all_preds)
    predicted = tokenizer.batch_decode(preds_concat, skip_special_tokens=True)

    result = metric.compute(predictions=predicted, references=summaries, use_stemmer=True)
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds_concat]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}

    if evaluate_single:
        result_single = defaultdict(list)
        result["lens"] = prediction_lens
        for i in range(len(predicted)):
            result2 = metric.compute(predictions=[predicted[i]], references=[summaries[i]], use_stemmer=True)
            result2 = {key: value.mid.fmeasure * 100 for key, value in result2.items()}
            result2 = {k: round(v, 4) for k, v in result2.items()}
            for key, val in result2.items():
                result_single[key].append(val)
        return result, result_single

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
    type=Path,
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
@click.option(
    "-f",
    "--filtered",
    type=bool,
    default=True,
    help="If process on filtered texts.",
)
@click.option(
    "-r",
    "--random",
    type=bool,
    default=False,
    help="If should select random sentences.",
)
@click.option(
    "-m",
    "--model-name",
    type=str,
    default="newsroom",
    help="Pegasus model name to use.",
)
@click.option(
    "--batch-size",
    type=int,
    default=4,
    help="Size of batch size.",
)
@click.option(
    "--device",
    type=str,
    default="cpu",
    help="Device to use, one of cpu, cuda.",
)
def main(
    input_json: Path,
    save_file: Optional[Path] = None,
    threshold: float = 0.01,
    filtered: bool = True,
    random: bool = False,
    model_name: str = "newsroom",
    batch_size: int = 4,
    device: str = "cpu"
):
    params = locals()

    # init model
    model_name = f"google/pegasus-{model_name}"
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)

    if random:
        texts, summaries, filtered_texts, n_sentences = load_random_as_many_as_filtered(input_json, threshold)
    else:
        texts, summaries, filtered_texts, n_sentences = load_filtered(input_json, threshold)

    if filtered:
        result, result_single = evaluate(filtered_texts, summaries, tokenizer, model, batch_size, device, True)
    else:
        result, result_single = evaluate(texts, summaries, tokenizer, model, batch_size, device, True)

    if save_file is not None:
        with open(save_file, "w") as f:
            json.dump(result, f, indent=4)

    logger = neptune.init(project=NEPTUNE_PROJECT, api_token=NEPTUNE_API_TOKEN)
    logger["parameters"] = params
    logger["results"] = result
    logger["result_single"] = result_single
    logger["mean_filtered_length"] = sum([len(t) for t in filtered_texts]) / len(filtered_texts)
    logger["mean_text_length"] = sum([len(t) for t in texts]) / len(filtered_texts)
    logger["filtered_lengths"] = [len(t) for t in filtered_texts]
    logger["text_lengths"] = [len(t) for t in texts]
    logger["method"] = "Pegasus"
    logger.stop()


if __name__ == '__main__':
    main()
