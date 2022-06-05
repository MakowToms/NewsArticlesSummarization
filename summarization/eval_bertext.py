import json
from pathlib import Path
from typing import Optional

import click
import neptune.new as neptune
import numpy as np
from datasets import load_metric
from tqdm import tqdm
from summarizer import Summarizer

from config import NEPTUNE_PROJECT, NEPTUNE_API_TOKEN
from subjectivity.filter_subjectivity import load_filtered, load_random_as_many_as_filtered

# init model
model = Summarizer()

# init metric
metric = load_metric("rouge")


# evaluate
def change_size(predictions, dim):
    new = np.zeros([predictions.shape[0], dim])
    new[:, :predictions.shape[1]] = predictions
    return new


def evaluate(texts, summaries, n_sentences, summary_length, summary_ratio, max_summary_length):

    predicted = []
    for text, n in tqdm(zip(texts, n_sentences)):
        if summary_ratio:
            num_sentences = int(np.ceil(summary_ratio * n))
            predicted.append(model(text, num_sentences=num_sentences))
        elif summary_length:
            predicted.append(model(text, num_sentences=summary_length))
        else:
            num_sentences = model.calculate_optimal_k(text, k_max=max_summary_length)
            predicted.append(model(text, num_sentences=num_sentences))

    result = metric.compute(predictions=predicted, references=summaries, use_stemmer=True)
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    prediction_lens = [len(pred) for pred in predicted]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}

    return result


@click.command(help="""
Script to generate Bert extractive summarizer summaries.

Example usage:
python -m summarization.eval_bertext -i data/newsroom/sample-v2_subj_scored_blob.json -l 5
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
    default=0.01,
    help="Threshold to filter texts.",
)
@click.option(
    "-l",
    "--summary-length",
    type=int,
    help="Number of sentences from text to choose as a summary if --summary-ratio is not given.",
)
@click.option(
    "--summary-ratio",
    type=float,
    help="Ratio of sentences from text to choose as a summary.",
)
@click.option(
    "-m",
    "--max-summary-length",
    type=int,
    default=20,
    help="Max summary length to estimate if --summary-length  or --summary-ratio is not set.",
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
def main(
    input_json: Path,
    save_file: Optional[Path] = None,
    threshold: float = 0.01,
    summary_length: Optional[int] = None,
    summary_ratio: Optional[float] = None,
    max_summary_length: int = 20,
    filtered: bool = True,
    random: bool = False,
):
    params = locals()
    if random:
        texts, summaries, filtered_texts, n_sentences = load_random_as_many_as_filtered(input_json, threshold)
    else:
        texts, summaries, filtered_texts, n_sentences = load_filtered(input_json, threshold)

    if filtered:
        result = evaluate(filtered_texts, summaries, n_sentences, summary_length, summary_ratio, max_summary_length)
    else:
        result = evaluate(texts, summaries, n_sentences, summary_length, summary_ratio, max_summary_length)

    if save_file is not None:
        with open(save_file, "w") as f:
            json.dump(result, f, indent=4)

    logger = neptune.init(project=NEPTUNE_PROJECT, api_token=NEPTUNE_API_TOKEN)
    logger["parameters"] = params
    logger["results"] = result
    logger["mean_filtered_length"] = sum([len(t) for t in filtered_texts]) / len(filtered_texts)
    logger["mean_text_length"] = sum([len(t) for t in texts]) / len(filtered_texts)
    logger["method"] = "BertExt"
    logger.stop()


if __name__ == '__main__':
    main()
