from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
RESULTS_DIR = Path("results/pegasus/")


def get_data_df():
    data = pd.concat([
        pd.read_csv(RESULTS_DIR / "news-summarization-NLP.csv"),
    ])
    data = data.drop_duplicates()
    data["is_blob"] = data["input_json"].apply(lambda x: x.endswith('blob.json'))
    data["is_multi"] = data["input_json"].apply(lambda x: x.startswith('data/multi') or x.startswith('data\multi'))
    return data


DF = get_data_df()
DF.columns

def plot_one(data, filter_type, plot_type="rouge1"):
    filtered = data[(data["random"] == False) & (data["filtered"] == True)]
    random = data[(data["random"] == True) & (data["filtered"] == True)]
    x1 = filtered["threshold"]
    x2 = random["threshold"]
    plt.plot(x1, filtered[plot_type], 'o-', label=f"Filtered by {filter_type}")
    plt.plot(x2, random[plot_type], 'o--', label="Random")
    if filter_type != "blob":
        plt.xscale("symlog", linthresh=10e-4)
    plt.legend()
    plt.ylabel(plot_type)
    plt.xlabel("Threshold")


def plot(is_blob, is_multi, method, title, filename):
    data = DF.loc[
        (DF["is_blob"] == is_blob)
        & (DF["is_multi"] == is_multi)
        & (DF["method"] == method)
    ]
    df2 = pd.DataFrame(data.loc[data["filtered"] == False])
    df2["threshold"] = 0
    df2["filtered"] = True
    data[data["filtered"] == False] = df2

    filter_type = "blob" if is_blob else "transformer"
    data = data.sort_values("threshold")

    plt.subplots(1, 3, figsize=(24, 6))
    plt.subplot(1, 3, 1)
    plt.title(title)
    plot_one(data, filter_type, "rouge1")
    plt.subplot(1, 3, 2)
    plt.title(title)
    plot_one(data, filter_type, "rouge2")
    plt.subplot(1, 3, 3)
    plt.title(title)
    plot_one(data, filter_type, "rougeL")
    plt.tight_layout()

    if filename is None:
        plt.show()
    else:
        plt.savefig(Path("plots") / (filename + ".png"))


plot(True, True, "Pegasus", "Pegasus with multinews", "pegasus_multi_blob")
plot(False, True, "Pegasus", "Pegasus with multinews", "pegasus_multi_transformer")


def plot2_one(data, plot_type="rouge1"):
    blob = data[(data["random"] == False) & (data["is_blob"] == True)]
    transformer = data[(data["random"] == False) & (data["is_blob"] == False)]
    random = data[(data["random"] == True)]
    plt.plot(blob['filtered_l'], blob[plot_type], 'o-', label="Filtered by blob")
    plt.plot(transformer['filtered_l'], transformer[plot_type], 'x-', label="Filtered by transformer")
    plt.plot(random['filtered_l'], random[plot_type], 'o--', label="Random")
    plt.legend()
    plt.ylabel(plot_type)
    plt.xlabel("Mean text length")


def plot2(is_multi, method, title, filename):
    data = DF.loc[
        (DF["is_multi"] == is_multi)
        & (DF["method"] == method)
    ]
    df2 = pd.DataFrame(data.loc[data["filtered"] == False])
    df2["threshold"] = 0
    df2["filtered"] = True
    df2["filtered_l"] = df2['text_len']
    data[data["filtered"] == False] = df2

    data = data.sort_values("threshold")

    plt.subplots(1, 3, figsize=(24, 6))
    plt.subplot(1, 3, 1)
    plt.title(title)
    plot2_one(data, "rouge1")
    plt.subplot(1, 3, 2)
    plt.title(title)
    plot2_one(data, "rouge2")
    plt.subplot(1, 3, 3)
    plt.title(title)
    plot2_one(data, "rougeL")
    plt.tight_layout()

    if filename is None:
        plt.show()
    else:
        plt.savefig(Path("plots") / (filename + ".png"))


plot2(True, "Pegasus", "Pegasus with multinews", "pegasus_multi")
