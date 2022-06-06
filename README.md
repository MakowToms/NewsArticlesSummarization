# NewsArticlesSummarization

This project researches the use of filtering subjective sentences in the task of news article summarization. Sentences in an article are assigned scores using one of the methods (Blob subjectivity score or sentiment analysis transformer model). Then sentences with a score under a given threshold can be removed before feeding the rest of the text to a summarization model (BertExt or Pegasus).

## Preparing data

Firstly, input data needs to be saved in a compatible json format. @TODO how?

## Scoring subjectivity

Before a summarization model is run, a json file with subjectivity scores needs to be generated. This can be done by running i.e.:

    python -m subjectivity.score_subjectivity -p data/newsroom/sample-v1.json -s blob
    
for blob or:

    python -m subjectivity.score_subjectivity -p data/newsroom/sample-v1.json -s transformer
    
for transformer

## Creating summaries

Summaries with BertExt can be generated using:

    python -m summarization.eval_bertext -i data/newsroom/sample-v2_subj_scored_blob.json -l 5
    
see help or script for list and explanation of available parameters. Analogically, summaries with Pegasus can be created using:

    python -m summarization.eval_pegasus -i data/newsroom/sample-v2_subj_scored_blob.json -t 0.4
