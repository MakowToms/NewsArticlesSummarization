import os

if "NEPTUNE_API_TOKEN" not in os.environ:
    raise EnvironmentError("Please set NEPTUNE_API_TOKEN environmental variable.")
NEPTUNE_API_TOKEN = os.environ.get("NEPTUNE_API_TOKEN")

if "NEPTUNE_PROJECT_NAME" in os.environ:
    NEPTUNE_PROJECT = os.environ.get("NEPTUNE_PROJECT_NAME")
else:
    NEPTUNE_PROJECT = "DSW-RedTurtle/news-summarization-NLP"
