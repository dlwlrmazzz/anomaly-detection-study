from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, random_split

def load_mnist(is_train=True):

    dataset = MNIST(
        '../data', train=is_train, download=True,
        transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize(mean=[0.5],
                                                    std=[0.5])])
    )

    return dataset

def get_loaders(config):
    # train, valid
    dataset = load_mnist(is_train=True)

    train_cnt = int(len(dataset) * config.train_ratio)
    valid_cnt = len(dataset) - train_cnt
    mnist_train, mnist_val = random_split(dataset, [train_cnt, valid_cnt])

    train_loader = DataLoader(
        dataset=mnist_train,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
    )
    valid_loader = DataLoader(
        dataset=mnist_val,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
    )
    # test
    dataset_test = load_mnist(is_train=False)
    test_loader = DataLoader(
        dataset=dataset_test,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=True,
    )

    return train_loader, valid_loader, test_loader


# class MnistDataset(Dataset):
    
#     def __init__(self, data, labels, flatten=True):
#         self.data = data
#         self.labels = labels
#         self.flatten = flatten

#         super().__init__()

#     def __len__(self):
#         return self.data.size(0)

#     def __getitem__(self, idx):
#         x = self.data[idx]
#         y = self.labels[idx]

#         if self.flatten:
#             x = x.view(-1)

#         return x, y