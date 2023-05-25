import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import pickle
    
    
    
def load_mnist(root):
    transform = transforms.Compose([transforms.ToTensor(),  
                                    transforms.Lambda(lambda x: torch.where(x < 0.5, -1., 1.))])
    trainset = datasets.MNIST(root, train=True, transform=transform, download=True)
    testset = datasets.MNIST(root, train=False, transform=transform, download=True)
    return trainset, testset


def load_news(root):
    # read data from pickle file
    with open(f"{root}/processed/cleaned_categories10.pkl", "rb") as f:
        data = pickle.load(f)
        x, y = data["x"].toarray(), data["y"]
        label_ids, vocab = data["label_ids"], data["vocab"]

    # binarize by thresholding 0
    x = torch.where((x > 0), torch.ones(x.size()), -torch.ones(x.size()))

    # split into sub-datasets
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    train, val, test = torch.utils.data.random_split(
        dataset,
        [
            round(0.8 * len(dataset)),
            round(0.1 * len(dataset)),
            len(dataset) - round(0.8 * len(dataset)) - round(0.1 * len(dataset)),
        ],
        torch.Generator().manual_seed(42),  # Use same seed to split data
    )
    return train, val, test, vocab, list(label_ids)
