import click
import sys
import json
from transformers import pipeline
import nltk
from textblob import TextBlob


@click.command(help="Create sentence scores for a json file with texts")
@click.option(
    "-p",
    "--path",
    default="data/newsroom/sample-v1.json",
    type=str,
    help="Path to json file with texts.",
)
@click.option(
    "-s",
    "--scorer",
    default="blob",
    type=str,
    help="Name of the scorer to use: [blob, transformer].",
)
def main(
        path: str,
        scorer: str,
):
    nltk.download('punkt')
    texts, summaries = load_json_data(path)
    scorers = {'blob': BlobSentiment, 'transformer': TransformerSentiment}
    scorer = scorers[scorer]()
    all_sentences = []
    all_scores = []
    for text in texts:
        sentences = nltk.tokenize.sent_tokenize(text)
        scores = scorer.score(sentences)
        all_sentences.append(sentences)
        all_scores.append(scores)
    save_json_data(path, texts, summaries, all_scores, all_sentences, scorer)
    return 0


class BlobSentiment:

    def score(self, sentences):
        sentiments = []
        for sentence in sentences:
            blob = TextBlob(sentence)
            sentiments.append(blob.sentiment.subjectivity)
        return [1-sentiment for sentiment in sentiments]


class TransformerSentiment:
    def __init__(self, task='sentiment-analysis'):
        self.pipeline = pipeline(task)

    def score(self, sentences):
        sentiments = self.pipeline(sentences)
        return [1-sentiment['score'] for sentiment in sentiments]


def load_json_data(path):
    with open(path) as f:
        text_json = json.load(f)
    texts = [i['text'] for i in text_json]
    summaries = [i['summary'] for i in text_json]
    return texts, summaries


def save_json_data(path, texts, summaries, scores, sentences, suffix=''):
    split_path = path.split('.')
    save_path = split_path[0] + f'_subj_scored_{suffix}.' + split_path[1]

    with open(save_path, "w") as f:
        json.dump([{
            "text": text,
            "summary": summary,
            "scores": score,
            "sentences": sentence,
        } for text, summary, score, sentence in zip(texts, summaries, scores, sentences)], f, indent=4)


if __name__ == '__main__':
    sys.exit(main())
