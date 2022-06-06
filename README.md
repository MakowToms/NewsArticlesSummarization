# NewsArticlesSummarization

This project researches the use of filtering subjective sentences in the task of news article summarization. Sentences in an article are assigned scores using one of the methods (Blob subjectivity score or sentiment analysis transformer model). Then sentences with a score under a given threshold can be removed before feeding the rest of the text to a summarization model (BertExt or Pegasus).

## Preparing data

Firstly, input data needs to be saved in a compatible json format. 
It can be done by using `save_sample_data` method from `summarization/data.py`. 
The sample data are saved for instance in `data/multinews/sample-v1.json`.

## Scoring subjectivity

Before a summarization model is run, a json file with subjectivity scores needs to be generated. This can be done by running i.e.:

    python -m subjectivity.score_subjectivity -p data/newsroom/sample-v1.json -s blob
    
for blob or:

    python -m subjectivity.score_subjectivity -p data/newsroom/sample-v1.json -s transformer
    
for transformer

The `-p` parameter have to provide path to json with sentences, and `-s` denotes the method to give an objectivity score.

## Creating summaries

Before running any script, you need to set up `NEPTUNE_API_TOKEN` environment variable with your neptune api token: login to https://neptune.ai/ and in menu go to `Get your API token`. Secondly you need to create any neptune project and set up `NEPTUNE_PROJECT_NAME` environment variable with `<your-neptune-login>/<your-project-name>`.

Summaries with BertExt can be generated using:

    python -m summarization.eval_bertext -i data/newsroom/sample-v2_subj_scored_blob.json -l 5
    
see help or script for list and explanation of available parameters. 
Analogically, summaries with Pegasus can be created using:

    python -m summarization.eval_pegasus -i data/newsroom/sample-v2_subj_scored_blob.json -t 0.4
    
In both methods the most important is parameter `-i` to file with ojectivity scores of sentences. 
Together with this the threshold `-t` is important to know which sentences to filter out. 
The bigger threshold filters more sentences.
