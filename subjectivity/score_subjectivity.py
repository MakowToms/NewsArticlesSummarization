import sys
import json
from transformers import pipeline
import nltk
nltk.download('punkt')

PATH = 'data/newsroom/sample-v2.json'


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


def save_json_data(path, texts, summaries, scores, sentences):
    split_path = path.split('.')
    save_path = split_path[0] + '_subj_scored.' + split_path[1]

    with open(save_path, "w") as f:
        json.dump([{
            "text": text,
            "summary": summary,
            "scores": score,
            "sentences": sentence,
        } for text, summary, score, sentence in zip(texts, summaries, scores, sentences)], f, indent=4)


def main():
    texts, summaries = load_json_data(PATH)
    scorer = TransformerSentiment()
    all_sentences = []
    all_scores = []
    for text in texts:
        sentences = nltk.tokenize.sent_tokenize(text)
        scores = scorer.score(sentences)
        all_sentences.append(sentences)
        all_scores.append(scores)
    save_json_data(PATH, texts, summaries, all_scores, all_sentences)
    return 0


if __name__ == '__main__':
    sys.exit(main())
