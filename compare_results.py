from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
RESULTS_DIR = Path("results/")
CSV_NAME = "news-summarization-NLP.csv"


def get_data_df():
    data = pd.concat([
        pd.read_csv(RESULTS_DIR / "pegasus" / CSV_NAME),
        pd.read_csv(RESULTS_DIR / "bert_ext_multi" / CSV_NAME),
        pd.read_csv(RESULTS_DIR / "bert_ext_room" / CSV_NAME),
    ])
    data = data.drop_duplicates()
    data["is_blob"] = data["input_json"].apply(lambda x: x.endswith('blob.json'))
    data["is_multi"] = data["input_json"].apply(lambda x: x.startswith('data/multi') or x.startswith('data\multi'))
    return data


DF = get_data_df()
DF = DF[DF["Id"].apply(lambda x: x not in ["NEWS-4", "NEWS-121", "NEWS-127", "NEWS-185", "NEWS-191"])]


def plot_one(data, filter_type, plot_type="rouge1"):
    filtered = data[(data["random"] == False) & (data["filtered"] == True)]
    random = data[(data["random"] == True) & (data["filtered"] == True)]
    x1 = filtered["threshold"]
    x2 = random["threshold"]
    color = "orange" if filter_type == "blob" else "blue"
    plt.plot(x1, filtered[plot_type], 'o-', color=color, label=f"Filtered by {filter_type}")
    plt.plot(x2, random[plot_type], 'x--', color="green", label="Random")
    if filter_type != "blob":
        plt.xscale("symlog", linthresh=10e-4)
    # plt.legend()
    plt.title(plot_type)
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

    plt.subplots(1, 3, figsize=(16, 4.2))
    plt.subplot(1, 3, 1)
    # plt.title(title)
    plot_one(data, filter_type, "rouge1")
    plt.subplot(1, 3, 2)
    # plt.title(title)
    plot_one(data, filter_type, "rouge2")
    plt.subplot(1, 3, 3)
    # plt.title(title)
    plot_one(data, filter_type, "rougeL")
    # plt.tight_layout()
    plt.suptitle(title)
    plt.legend(title='Filtering method')

    if filename is None:
        plt.show()
    else:
        plt.savefig(Path("plots") / (filename + ".png"))


plot(True, True, "Pegasus", "Pegasus on multinews dataset", "pegasus_multi_blob")
plot(False, True, "Pegasus", "Pegasus on multinews dataset", "pegasus_multi_transformer")
plot(True, False, "Pegasus", "Pegasus on newsroom dataset", "pegasus_room_blob")
plot(False, False, "Pegasus", "Pegasus on newsroom dataset", "pegasus_room_transformer")
plot(True, True, "BertExt", "BertExt on multinews dataset", "bertExt_multi_blob")
plot(False, True, "BertExt", "BertExt on multinews dataset", "bertExt_multi_transformer")
plot(True, False, "BertExt", "BertExt on newsroom dataset", "bertExt_room_blob")
plot(False, False, "BertExt", "BertExt on newsroom dataset", "bertExt_room_transformer")


def plot2_one(data, plot_type="rouge1"):
    blob = data[(data["random"] == False) & (data["is_blob"] == True)]
    transformer = data[(data["random"] == False) & (data["is_blob"] == False)]
    random = data[(data["random"] == True)]
    plt.plot(blob['filtered_l'], blob[plot_type], 'o-', color="orange", label="Filtered by blob")
    plt.plot(transformer['filtered_l'], transformer[plot_type], 'o-', color="blue", label="Filtered by transformer")
    plt.plot(random['filtered_l'], random[plot_type], 'x--', color="green", label="Random")
    # plt.legend()
    plt.title(plot_type)
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

    data = data.sort_values("filtered_l")

    plt.subplots(1, 3, figsize=(16, 4.2))
    plt.subplot(1, 3, 1)
    # plt.title(title)
    plot2_one(data, "rouge1")
    plt.subplot(1, 3, 2)
    # plt.title(title)
    plot2_one(data, "rouge2")
    plt.subplot(1, 3, 3)
    # plt.title(title)
    plot2_one(data, "rougeL")
    # plt.tight_layout()
    plt.suptitle(title)
    plt.legend(title='Filtering method')

    if filename is None:
        plt.show()
    else:
        plt.savefig(Path("plots") / (filename + ".png"))


plot2(True, "Pegasus", "Pegasus on multinews dataset", "pegasus_multi")
plot2(False, "Pegasus", "Pegasus on newsroom dataset", "pegasus_room")
plot2(True, "BertExt", "BertExt on multinews dataset", "bertExt_multi")
plot2(False, "BertExt", "BertExt on newsroom dataset", "bertExt_room")


df = DF.set_index("Id").loc[["NEWS-74", "NEWS-202", "NEWS-184",  # multi bert
                        "NEWS-75", "NEWS-48",  # pegasus multi
                        "NEWS-30", "NEWS-96",  # newsroom bert
                        "NEWS-172", "NEWS-8"], :]  # newsroom multi
df = df.loc[
:, ["rouge1", "rouge2", "rougeL", "filtered_l", "text_len", "threshold", "is_blob", "is_multi", "method", "filtered"]
]
df["filtering"] = df.apply(lambda x: ("Blob" if x["is_blob"] else "Transformer") + " " + str(x["threshold"]) if x["filtered"] else "Baseline", axis=1)
df["length"] = df.apply(lambda x: x["filtered_l"] if x["filtered"] else x["text_len"], axis=1)
df["dataset"] = df.apply(lambda x: "Multinews" if x["is_multi"] else "Newsroom", axis=1)
df = df.loc[
    :, ["dataset", "method", "filtering", "length", "rouge1", "rouge2", "rougeL"]
]

df.to_latex(Path("plots") / "results_table.tex", index=False)

