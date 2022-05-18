import json
import numpy as np


def load_json_scored(path):
    with open(path) as f:
        text_json = json.load(f)
    texts = [i['texts'] for i in text_json]
    summaries = [i['summary'] for i in text_json]
    sentences = [i['sentences'] for i in text_json]
    scores = [i['scores'] for i in text_json]
    return texts, summaries, sentences, scores


def load_filtered(path, threshold=0.001):
    texts, summaries, all_sentences, all_scores = load_json_scored(path)
    filtered_texts = []
    for sentences_list, scores_list in zip(all_sentences, all_scores):
        filtered_text_list = []
        for sentences, scores in zip(sentences_list, scores_list):
            sentences_np = np.array(sentences)
            filtered_sentences = sentences_np[
                np.where(np.array(scores, dtype=float) > threshold)
            ]
            filtered_text_list.append(' '.join(filtered_sentences))
        filtered_texts.append(filtered_text_list)
    return texts, summaries, filtered_texts
