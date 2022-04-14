import torch
from datasets import load_dataset, DatasetDict

# force not using cuda
# torch.cuda.is_available = lambda : False


def load_data(dataset_name, is_subset=True):
    if dataset_name == "newsroom":
        return load_newsroom(subset=is_subset)


def load_newsroom(subset=True):
    if subset:
        dataset = load_dataset("newsroom", data_dir="data/newsroom", split="train[0:10000]")

        # 90% train, 10% test + validation
        train_testvalid = dataset.train_test_split(test_size=0.1)

        # Split the 10% test + valid in half test, half valid
        test_valid = train_testvalid['test'].train_test_split(test_size=0.5)

        # gather everyone if you want to have a single DatasetDict
        dataset = DatasetDict({'train': train_testvalid['train'], 'test': test_valid['test'], 'validation': test_valid['train']})
    else:
        dataset = load_dataset("newsroom", data_dir="data/newsroom")
    return dataset
