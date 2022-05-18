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


def join_texts(texts):
    return [
        ' ||||| '.join(text_list)
        for text_list in texts
    ]


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
        filtered_texts.append(' ||||| '.join(filtered_text_list))
    return join_texts(texts), summaries, filtered_texts


def load_random_as_many_as_filtered(path, threshold=0.001):
    texts, summaries, all_sentences, all_scores = load_json_scored(path)
    filtered_texts = []
    for sentences_list, scores_list in zip(all_sentences, all_scores):
        filtered_text_list = []
        for sentences, scores in zip(sentences_list, scores_list):
            sentences_np = np.array(sentences)
            n_sentences = np.sum(np.array(scores, dtype=float) > threshold)
            if n_sentences > 0:
                random_sentences = np.random.choice(np.arange(0, sentences_np.shape[0]), n_sentences, replace=False)
                filtered_sentences = sentences_np[random_sentences]
                filtered_text_list.append(' '.join(filtered_sentences))
            else:
                filtered_text_list.append('')
        filtered_texts.append(' ||||| '.join(filtered_text_list))
    return join_texts(texts), summaries, filtered_texts
