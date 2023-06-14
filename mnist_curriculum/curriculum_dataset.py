import numpy as np
import torch as th
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, Sampler
import torch
import torchvision
import tqdm
from transformers import ViTForImageClassification
import matplotlib.pyplot as plt
import random
import pickle


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

class MNISTDifficultyClassifier:
    def __init__(self, model, mnist: MNIST):
        self.model = model
        self.classes = th.arange(0, 10)
        self.neu_label = th.ones(10) / 10 # neutral
        self.model.eval()
        self.device = 'cpu'
        if th.cuda.is_available():
            self.device = 'cuda'
            self.model.to(self.device)
        self.difficulties = self.assign_difficulties(mnist) # difficulties[idx] = (label, difficulty)

    def assign_difficulties(self, mnist):
        difficulties = {}
        for idx in tqdm.tqdm(range(len(mnist)), total=len(mnist)):
            image, label = mnist[idx]
            image = image.repeat(3, 1, 1)
            pil_image = torchvision.transforms.ToPILImage()(image)
            resized_image = torchvision.transforms.Resize((224, 224))(pil_image)
            resized_image = torchvision.transforms.ToTensor()(resized_image)
            image = resized_image.unsqueeze(0).to(self.device)

            with th.no_grad():
                outputs = self.model(image)
                probabilities = th.nn.functional.softmax(outputs.logits, dim=1)
                max_prob, _ = th.max(probabilities, dim=1)
                difficulty = 1 / max_prob.item()
                difficulties[idx] = (label, difficulty)

        return difficulties


    def save(self, path, _type='npy'):
        if _type == 'npy':
            np.save(path, self.difficulties)
        elif _type == 'csv':
            with open(path, 'w') as f:
                for idx, (label, difficulty) in self.difficulties.items():
                    f.write(f'{idx},{label},{difficulty}\n')
        elif _type == 'pkl':
            with open(path, 'wb') as f:
                pickle.dump(self.difficulties, f)
        else:
            raise NotImplementedError(f'Unknown type {_type}')




class MNISTCurriculum(object):
    def __init__(self, dataset: MNIST, difficulties=None):
        self.mnist = dataset
        self.neu_label = th.ones(10) / 10 # neutral
        self.classes = th.arange(0, 10)
        self.difficulties = difficulties

    def display_difficulty_spread(self):
        plt.hist([difficulty for _, difficulty in self.difficulties.values()])
        plt.show()

    def set_difficulties(self, difficulties):
        self.difficulties = difficulties


    def get_difficulty_index_based_on_distribution(self, distribution):
        difficulties = [(idx, t[1]) for idx, t in self.difficulties.items()]
        difficulties.sort(key=lambda x: x[1])
        difficulties = [idx for idx, _ in difficulties]
        easy_idx = difficulties[:int(len(difficulties) * distribution[0])]
        medium_idx = difficulties[int(len(difficulties) * distribution[0]):int(len(difficulties) * (distribution[0] + distribution[1]))]
        hard_idx = difficulties[int(len(difficulties) * (distribution[0] + distribution[1])):]

        return easy_idx, medium_idx, hard_idx

    def get_difficulty_index_based_on_thresholds(self, thresholds=(1.5, 5)):
        easy_idx = []
        medium_idx = []
        hard_idx = []
        for idx, t in self.difficulties.items():
            if t[1] < thresholds[0]:
                easy_idx.append(idx)
            elif t[1] < thresholds[1]:
                medium_idx.append(idx)
            else:
                hard_idx.append(idx)
        return easy_idx, medium_idx, hard_idx

    def __getitem__(self, idx):
            image, label = self.mnist[idx]
            pos_label = th.tensor(label)
            neg_label = self.classes[self.classes != label][th.randint(0, 9, (1,))].squeeze()
            return image, (
                th.nn.functional.one_hot(pos_label, num_classes=10),
                th.nn.functional.one_hot(neg_label, num_classes=10),
                self.neu_label.clone(),
            )

    def __len__(self):
        return len(self.mnist)



def load(path, _type='pkl'):
    return pickle.load(open(path, 'rb'))


class BalancedBatchSampler(Sampler):

    def __init__(self, easy_idx, medium_idx, hard_idx, batch_size, ratios, used_indices=None):
        self.indices = [easy_idx, medium_idx, hard_idx]
        self.batch_size = batch_size
        self.ratios = ratios
        self.used_indices = set() if used_indices is None else used_indices

    def _get_indices(self, indices, n):
        available_indices = list(set(indices) - self.used_indices)
        # print(f"Available indices: {len(available_indices)}, indices: {self.indices}, used: {self.used_indices}")
        if len(available_indices) < n:
            self.used_indices = set()
            available_indices = indices
        selected_indices = random.sample(available_indices, n)
        self.used_indices.update(selected_indices)
        return selected_indices

    def __iter__(self):
        self.current_indices = []
        for indices, ratio in zip(self.indices, self.ratios):
            self.current_indices += self._get_indices(indices, int(self.batch_size * ratio))
        remaining = self.batch_size - len(self.current_indices)
        for indices in self.indices:
            if remaining <= 0:
                break
            extra_indices = self._get_indices(indices, remaining)
            self.current_indices += extra_indices
            remaining -= len(extra_indices)
        random.shuffle(self.current_indices)

        for index in self.current_indices:
            yield index

        self.used_indices = set()

    def __len__(self):
        return max(len(indices) for indices in self.indices) // self.batch_size

def collate_fn(batch):
    #print(batch)
    images = []
    pos_labels = []
    neg_labels = []
    neu_labels = []
    for b in batch:
        images.append(b[0])
        pos_labels.append(b[1][0])
        neg_labels.append(b[1][1])
        neu_labels.append(b[1][2])
    return th.stack(images), (th.stack(pos_labels), th.stack(neg_labels), th.stack(neu_labels))

def get_sampler(easy_idx, medium_idx, hard_idx, batch_size, ratios):
    return BalancedBatchSampler(easy_idx, medium_idx, hard_idx, batch_size, ratios)


def get_balanced_dataloader(dataset, batch_size, sampler, num_workers=0):
    return DataLoader(
        dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, collate_fn=collate_fn,
    )

def adjust_ratios(epoch):
    # if epoch < 50:
    #     return [0.3, 0.3, 0.4]
    # elif epoch < 85:
    #     return [0.2, 0.4, 0.4]
    # else:
    #     return [0.1, 0.4, 0.5]
    if epoch < 50:
        return [0.1, 0.4, 0.5]
    elif epoch < 85:
        return [0.2, 0.4, 0.4]
    else:
        return [0.3, 0.3, 0.4]


if __name__ == '__main__':
    # test the sampler
    mnist = MNIST('./data', train=False, download=True, transform=transforms.ToTensor())
    # take only 100 samples
    mnist = th.utils.data.Subset(mnist, list(range(80)))
    mnist_curriculum = MNISTCurriculum(mnist)
    easy_idx, medium_idx, hard_idx = mnist_curriculum.get_difficulty_index_based_on_distribution([0.3, 0.3, 0.4])

    # print(len(easy_idx), len(medium_idx), len(hard_idx))


    sampler = BalancedBatchSampler(easy_idx, medium_idx, hard_idx, 8, [0.3, 0.3, 0.4])
    dataloader = DataLoader(mnist_curriculum, sampler=sampler)
    # print(dataloader)


    for batch in dataloader:
        print(batch)
        print(len(batch))
        print(batch[0].shape)

        break
