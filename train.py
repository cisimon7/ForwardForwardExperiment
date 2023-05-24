import hydra
import torch as th
from layers import get_model
from dataset import get_data
from tqdm.rich import trange
from utils import seed_devices
from omegaconf import DictConfig
from torch.utils.tensorboard import SummaryWriter


TRAIN_NAME = "train_01"


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
        optimizer: th.optim.Optimizer, test_loader, train_loader, test=False):
    
    model_ff.to(device), head.to(device)
    writer = SummaryWriter('logs/.tensorboard')
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
            loss = sum(layer_loss) + head_loss
            loss.backward()
            optimizer.step()
            
            layer_losses.append(layer_loss)
            head_losses.append(head_loss)
            
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
            writer.add_scalar(f"{TRAIN_NAME}/accuracy", accuracy, epoch)
        

@hydra.main(version_base=None, config_path="conf", config_name="config")
def train(cfg: DictConfig) -> None:
    seed_devices(cfg.seed)
    device = cfg.device
    
    train_set, test_set, train_loader, test_loader = get_data(
        cfg.data.batch_size, cfg.data.num_workers
    )
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
        train_loader, test_loader,
        cfg.train.test_model
    )
    
    if cfg.train.save_model:
        th.save(model_ff.state_dict(), f"./models/{TRAIN_NAME}/model_ff")
        th.save(head.state_dict(), f"./models/{TRAIN_NAME}/head")
        

if __name__ == "__main__":
    train()
            