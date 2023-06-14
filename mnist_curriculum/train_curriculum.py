import hydra
import torch as th
from layers import get_model
from curriculum_dataset import *
import tqdm as tqdm
from utils import seed_devices
from omegaconf import DictConfig
from torch.utils.tensorboard import SummaryWriter
from dataset import get_data as gd
import os

# Train Notes
# train01: batch size 100
# train02: kaiming init
# train03: adjust ratios included -> easy 50 -> medium 85 -> hard
# train04: adjust ratios reversed

TRAIN_NAME = "mnist_train_curriculum_00"
writer = SummaryWriter('logs/.tensorboard')

#
def test_model(
        device,
        model_ff: th.nn.Module, head: th.nn.Module):
    model_ff.to(device), head.to(device)
    scores = []

    _, _, _, test_loader = gd()

    for (images, (pos_label, neg_label, neu_label)) in test_loader:
        images, pos_label = images.to(device), pos_label.to(device)
        neg_label, neu_label = neg_label.to(device), neu_label.to(device)

        neu_data = th.cat([images.flatten(start_dim=1), neu_label], dim=1)
        neu_data = th.nn.functional.layer_norm(neu_data, neu_data.size())

        model_ff.eval()
        with th.no_grad():
            out = model_ff(neu_data, neu_label)
        out = head(out.detach())

        prediction = out.argmax(dim=1, keepdims=True)
        actual = pos_label.argmax(dim=1, keepdims=True)

        accuracy = (prediction == actual).sum() / actual.size(dim=0)
        scores.append(accuracy)

    return th.stack(scores).mean()

def train_pass(
        device, epochs,
        classification_loss_func,
        model_ff: th.nn.Module, head: th.nn.Module,
        optimizer: th.optim.Optimizer, test=True, num_workers=0, batch_size=None):
    model_ff.to(device), head.to(device)
    total_loss = 0
    batch_size = 100
    epochs = 100

    train_set, test_set = get_mnist()
    # train_set = th.utils.data.Subset(train_set, range(50))
    train_set = th.utils.data.Subset(train_set, range(50000))
    # test_set = th.utils.data.Subset(test_set, range(5))

    if os.path.exists("train_curriculum.pkl"):
        train_difficulties = load("train_curriculum.pkl", _type='pkl')
        train_curriculum = MNISTCurriculum(train_set, train_difficulties)
    else:
        train_curriculum = get_difficulty_classifier(
            train_set, "train_curriculum.pkl"
        )
    if os.path.exists("test_curriculum.pkl"):
        test_difficulties = load("test_curriculum.pkl", _type='pkl')
        test_curriculum = MNISTCurriculum(test_set, test_difficulties)
    else:
        test_curriculum = get_difficulty_classifier(
            test_set, "test_curriculum.pkl"
        )
    distribution = (0.5, 0.3, 0.2)
    r = (0.6, 0.3, 0.1)
    train_dist = train_curriculum.get_difficulty_index_based_on_distribution(distribution)
    easy_train_set, medium_train_set, hard_train_set = train_dist
    # sampler = get_sampler(easy_train_set, medium_train_set, hard_train_set, batch_size=batch_size, ratios=r)
    # dataloader = get_balanced_dataloader(train_curriculum, batch_size, sampler, num_workers=num_workers)

    # Generate a unique fixed pattern for each class
    for epoch in tqdm.tqdm(range(epochs)):
        layer_losses, head_losses = [], []
        print(f"Epoch: {epoch}")
        ratios = adjust_ratios(epoch)
        if ratios != r:
            sampler = get_sampler(easy_train_set, medium_train_set, hard_train_set, batch_size=batch_size, ratios=ratios)
            dataloader = get_balanced_dataloader(train_curriculum, batch_size, sampler, num_workers=num_workers)
            r = ratios

        for (images, (pos_label, neg_label, neu_label)) in dataloader:
            optimizer.zero_grad()
            images, pos_label = images.to(device), pos_label.to(device)
            neg_label, neu_label = neg_label.to(device), neu_label.to(device)

            pos_data = th.cat([images.flatten(start_dim=1), pos_label], dim=1)
            neg_data = th.cat([images.flatten(start_dim=1), neg_label], dim=1)
            neu_data = th.cat([images.flatten(start_dim=1), neu_label], dim=1)

            pos_data = th.nn.functional.layer_norm(pos_data, pos_data.size())
            neg_data = th.nn.functional.layer_norm(neg_data, neg_data.size())
            neu_data = th.nn.functional.layer_norm(neu_data, neu_data.size())

            data = th.cat([pos_data, neg_data])
            label = th.cat([
                th.ones((pos_data.size(dim=0), 1), device=device),
                th.zeros((neg_data.size(dim=0), 1), device=device)
            ])

            model_ff.train()

            model_ff(data, label)

            layer_loss = model_ff.losses()
            model_ff.eval()
            with th.no_grad():
                out = model_ff(neu_data, neu_label)
            out = head(out.detach())
            head_loss = classification_loss_func(pos_label.float(), out.float())
            loss = sum(layer_loss) + head_loss
            loss.backward()
            optimizer.step()

            layer_losses.append(layer_loss)
            head_losses.append(head_loss)

        if epoch % 5 == 0:
            print(f"Epoch: {epoch}, Loss: {loss.item()}, Head Loss: {head_loss.item()}, Layer Loss: {sum(layer_loss)}, Total Loss: {total_loss}")

        layer_losses = th.Tensor(layer_losses).T.mean(dim=1)

        writer.add_scalars(
            f"{TRAIN_NAME}/loss_layers",
            {f"layer_{i}": layer_losses[i].item() for (i, val) in enumerate(layer_losses)},
            epoch
        )
        writer.add_scalar(f"{TRAIN_NAME}/loss_head", th.sum(th.vstack(head_losses)).item(), epoch)

        if test:
            with th.no_grad():
                accuracy = test_model(device, model_ff, head)
            writer.add_scalar(f"{TRAIN_NAME}/accuracy", accuracy, epoch)


def get_difficulty_classifier(dataset, path):
    train_difficulty_classifier = MNISTDifficultyClassifier(
        ViTForImageClassification.from_pretrained(
            'google/vit-base-patch16-224-in21k'
        ),
        dataset,
    )
    train_difficulty_classifier.save(path, _type='pkl')
    return MNISTCurriculum(dataset, train_difficulty_classifier.difficulties)

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def train(cfg: DictConfig) -> None:
    seed_devices(cfg.seed)
    device = cfg.device

    loss_func = th.nn.CrossEntropyLoss()
    model_ff, head = get_model(cfg)

    optimizer = th.optim.SGD(
        [
            dict(
                params=model_ff.parameters(),
                lr=cfg.train.lr,
                weight_decay=cfg.train.lamda,
                momentum=0
            ),
            dict(
                params=head.parameters(),
                lr=cfg.train.head_lr,
                weight_decay=cfg.train.head_lamda,
                momentum=0
            )
        ]
    )

    train_pass(
        device, cfg.train.epochs, loss_func,
        model_ff, head, optimizer,
        # train_loader, test_loader,
        cfg.train.test_model, #cfg.train.num_workers, cfg.train.batch_size
    )

    # if cfg.train.save_model:
    if not os.path.exists(f"../models/{TRAIN_NAME}"):
        os.mkdir(f"../models/{TRAIN_NAME}")
    th.save(model_ff.state_dict(), f"../models/{TRAIN_NAME}/model_ff")
    th.save(head.state_dict(), f"../models/{TRAIN_NAME}/head")


if __name__ == '__main__':
    train()