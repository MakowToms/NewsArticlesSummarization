import json

from datasets import load_dataset, DatasetDict

# force not using cuda
# torch.cuda.is_available = lambda : False


def load_data(dataset_name, is_subset=True):
    if dataset_name == "newsroom":
        return load_newsroom(subset=is_subset)


def load_newsroom(subset=True):
    if subset:
        dataset = load_dataset("newsroom", data_dir="data/newsroom", split="test[0:10000]")

        # 90% train, 10% test + validation
        train_testvalid = dataset.train_test_split(test_size=0.1)

        # Split the 10% test + valid in half test, half valid
        test_valid = train_testvalid['test'].train_test_split(test_size=0.5)

        # gather everyone if you want to have a single DatasetDict
        dataset = DatasetDict({'train': train_testvalid['train'], 'test': test_valid['test'], 'validation': test_valid['train']})
    else:
        dataset = load_dataset("newsroom", data_dir="data/newsroom")
    return dataset


def filter_texts(texts):
    return [
        txt.replace("\n", "").replace("\x97", "") for txt in texts
    ]


def save_sample_data(dataset_name="newsroom", file_name="sample-v2", n_samples=100):
    newsroom = load_newsroom(subset=True)
    src_texts = filter_texts([str(text) for text in newsroom.data["train"]["text"][:n_samples]])
    target_texts = filter_texts([str(text) for text in newsroom.data["train"]["summary"][:n_samples]])

    mean_len = sum([len(t) for t in src_texts]) / n_samples
    mean_sum = sum([len(t) for t in target_texts]) / n_samples
    print(f"Mean text length: {mean_len}")
    print(f"Mean summary length: {mean_sum}")

    with open(f"data/{dataset_name}/{file_name}.json", "w") as f:
        json.dump([{
            "text": text,
            "summary": pred
        } for text, pred in zip(src_texts, target_texts)], f, indent=4)


def load_sample_data(dataset_name="newsroom", file_name="sample-v1"):
    with open(f"data/{dataset_name}/{file_name}.json", "r") as f:
        data = json.load(f)
    return data


# example usage
# multi_news = load_dataset("multi_news")
# newsroom = load_newsroom(subset=True)
