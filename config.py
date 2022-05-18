import os

if "NEPTUNE_API_TOKEN" not in os.environ:
    raise EnvironmentError("Please set NEPTUNE_API_TOKEN environmental variable.")
NEPTUNE_API_TOKEN = os.environ.get("NEPTUNE_API_TOKEN")

NEPTUNE_PROJECT = "DSW-RedTurtle/news-summarization-NLP"
