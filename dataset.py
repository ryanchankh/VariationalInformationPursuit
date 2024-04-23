import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset
import numpy as np
import pickle
import pandas as pd
from PIL import Image
    
    
def load_mnist(root):
    transform = transforms.Compose([transforms.ToTensor(),  
                                    transforms.Lambda(lambda x: torch.where(x < 0.5, -1., 1.))])
    trainset = datasets.MNIST(root, train=True, transform=transform, download=True)
    testset = datasets.MNIST(root, train=False, transform=transform, download=True)
    return trainset, testset


def load_news(root):
    # read data from pickle file
    with open(f"{root}/cleaned_categories10.pkl", "rb") as f:
        data = pickle.load(f)
        x, y = data["x"].toarray(), data["y"]
        label_ids, vocab = data["label_ids"], data["vocab"]

    # binarize by thresholding 0
    x = np.where((x > 0), np.ones(x.shape), -np.ones(x.shape))
    x = np.float32(x)

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


class CUB200(Dataset):
    """
    Returns a compatible Torch Dataset object customized for the CUB dataset
    """

    def __init__(
        self,
        root,
        image_dir='CUB_200_2011',
        split='train',
        transform=None,
):
        
        self.root = root
        self.image_dir = os.path.join(self.root, 'CUB', image_dir)
        self.transform = transform

        ## Image
        pkl_file_path = os.path.join(self.root, 'CUB', f'{split}class_level_all_features.pkl')
        self.data = []
        with open(pkl_file_path, "rb") as f:
            self.data.extend(pickle.load(f))
            
        ## Classes
        self.classes = pd.read_csv(os.path.join(self.image_dir, 'classes.txt'))['idx species'].values


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        _dict = self.data[idx]

        # image
        img_path = _dict['img_path']
        _idx = img_path.split("/").index("CUB_200_2011")
        img_path = os.path.join(self.root, 'CUB/CUB_200_2011', *img_path.split("/")[_idx + 1 :])
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        # class label
        class_label = _dict["class_label"]
        return img, class_label


def load_cub(root):    
    transform = transforms.Compose(
        [
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[2, 2, 2])
        ]
    )
    trainset = CUB200(root, image_dir='CUB_200_2011', split='train', transform=transform)
    testset = CUB200(root, image_dir='CUB_200_2011', split='test', transform=transform)
    valset = CUB200(root, image_dir='CUB_200_2011', split='val', transform=transform)
    return trainset, valset, testset
    
def load_cifar10(root):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = datasets.CIFAR10(
        root=root, train=True, download=True, transform=transform_train)
    testset = datasets.CIFAR10(
        root=root, train=False, download=True, transform=transform_test)
    return trainset, testset
