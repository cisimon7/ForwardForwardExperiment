import sys
import hydra
import torch as th
from tqdm.rich import trange
from utils import seed_devices
from cifar10.dataset import get_data
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from layers import get_model, SequentialFF, LinearLayerFF


TRAIN_NAME = "cifar10_fcn_theta_10000_2.5"
writer = SummaryWriter('logs/.tensorboard_final')


def test_model(
        device,
        model_ff: th.nn.Module, head: th.nn.Module, loader):

    model_ff.to(device), head.to(device)
    scores = []

    for (images, (pos_label, neg_label, neu_label)) in loader:
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
        optimizer: th.optim.Optimizer, scheduler, test_loader, train_loader, test=False):
    
    model_ff.to(device), head.to(device)
    last_accuracy = 0
    total_loss = 0
    
    for epoch in trange(epochs):
        layer_losses, head_losses = [], []
        for (images, (pos_label, neg_label, neu_label)) in test_loader:
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
            loss = th.mean(th.hstack(layer_loss + [head_loss]))
            loss.backward()
            optimizer.step()
            
            layer_losses.append(layer_loss)
            head_losses.append(head_loss)
            
        scheduler.step()
        layer_losses = th.Tensor(layer_losses).T.mean(dim=1)
            
        writer.add_scalars(
            f"{TRAIN_NAME}/loss_layers",
            {f"layer_{i}": layer_losses[i].item() for (i, val) in enumerate(layer_losses)},
            epoch
        )
        writer.add_scalar(f"{TRAIN_NAME}/loss_head", th.sum(th.vstack(head_losses)).item(), epoch)
        
        if test:
            with th.no_grad():
                accuracy = test_model(device, model_ff, head, train_loader)
                if cfg.train.save_on_improve and last_accuracy < accuracy:
                    last_accuracy = accuracy
                    th.save(model_ff.state_dict(), f"models/cifar10/fcn/model_ff_best")
                    th.save(head.state_dict(), f"models/cifar10/fcn/head_best")
            writer.add_scalar(f"{TRAIN_NAME}/accuracy", accuracy, epoch)
        

if __name__ == "__main__":
    with initialize(version_base=None, config_path="../../conf"):
        cfg = compose(config_name="config", overrides=["model=cifar10", "train=cifar10", "model/type=fcn"])
        seed_devices(cfg.seed)
        device = cfg.device
    
        train_set, test_set, train_loader, test_loader = get_data(
            32, cfg.model.type.batch_size, cfg.model.type.num_workers
        )
        loss_func = th.nn.MSELoss()
        model_ff = SequentialFF(
            LinearLayerFF(
                (3*32*32) + 10, 1_000,
                cfg.model.type.theta,
                ratio_peer=cfg.model.type.peer_ratio,
                device=cfg.device
            ),
            *(
                LinearLayerFF(
                    1_000, 1_000,
                    cfg.model.type.theta,
                    ratio_peer=cfg.model.type.peer_ratio,
                    device=cfg.device
                )
                for _ in th.arange(start=1, end=cfg.model.type.num_layers)
            )
        )
        head = th.nn.Sequential(
            th.nn.Linear((cfg.model.type.num_layers - 1) * 1_000, 1_000, bias=True),
            *(
                th.nn.Linear(1_000, 1_000, bias=True)
                for _ in th.arange(start=0, end=1)
            ),
            th.nn.Linear(1_000, 10, bias=False),
            # th.nn.Softmax(dim=1)
        )
        th.nn.init.zeros_(head[0].weight)
        
        optimizer = th.optim.AdamW([
            dict(
                params=model_ff.parameters(),
                lr=cfg.train.lr,
                weight_decay=cfg.train.lamda,
                # momentum=cfg.train.momentum 
            ),
            dict(
                params=head.parameters(),
                lr=cfg.train.head_lr,
                weight_decay=cfg.train.head_lamda,
                # momentum=cfg.train.momentum
            )
        ])
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
        train_pass(
            device, cfg.train.epochs, loss_func,
            model_ff, head, optimizer, scheduler,
            train_loader, test_loader,
            cfg.train.test_model
        )

        if cfg.train.save_model:
            th.save(model_ff.state_dict(), f"../models/{TRAIN_NAME}/model_ff")
            th.save(head.state_dict(), f"../models/{TRAIN_NAME}/head")
        
            