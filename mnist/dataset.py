import torch as th
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, Dataset


def get_mnist():
    train_set = MNIST(
        root='../datasets',
        train=True,                         
        transform=transforms.Compose([
            transforms.ToTensor(),
        ]),
        download=True,            
    )
    
    test_set = MNIST(
        root='../datasets', 
        train=False, 
        transform=transforms.Compose([
            transforms.ToTensor(),
        ])
    )
    return train_set, test_set


def get_data(batch_size=100, num_workers=0):
    
    train_set, test_set = get_mnist()
    train_set, test_set = MNISTData(train_set), MNISTData(test_set)
    
    train_loader = DataLoader(
        train_set, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_set, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )
    
    return train_set, test_set, train_loader, test_loader


class MNISTData(Dataset):
    def __init__(self, mnist: Dataset):
        self.mnist = mnist
        self.neu_label = th.ones(10) / 10
        self.classes = th.arange(0, 10)
        
    def __getitem__(self, idx):
        image, label = self.mnist[idx]
        pos_label = th.tensor(label)
        neg_label = self.classes[self.classes != label][th.randint(0, 9, (1,))].squeeze()
        return image, (
            th.nn.functional.one_hot(pos_label, num_classes=10),
            th.nn.functional.one_hot(neg_label, num_classes=10),
            self.neu_label
        )
    
    def __len__(self):
        return len(self.mnist)
        
    