import sys
import torch as th
from tqdm.rich import trange
from layers import SequentialFF
from omegaconf import DictConfig
from torch.utils.data import Subset
from cifar10.dataset import get_data
from hydra import compose, initialize
from cifar10.viT.utils import plot_patches
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from modules import PatchMakerModule, PatchEncoderFF, TransformerBlockFF, mlp_block


# sample = Subset(train_set, indices=[th.randint(low=0, high=len(train_set), size=(1,))])
# patches = model(sample[0][0][None, :])[0]
# plot_patches(sample[0][0], patches, patch_size)


TRAIN_NAME = "cifar10_vit_train_05"
writer = SummaryWriter('logs/.tensorboard2')


if __name__ == "__main__":
    with initialize(version_base=None, config_path="../../conf"):
        cfg = compose(config_name="config", overrides=["model=cifar10", "train=cifar10"])
        
        epochs = cfg.train.epochs
        transformer_layers = cfg.model.type.transformer_layer
        device, num_heads = cfg.device, cfg.model.type.num_heads
        image_size, patch_size = cfg.model.type.image_size, cfg.model.type.patch_size
        
        num_patches = (image_size // patch_size) ** 2
        projection_dim = cfg.model.type.projection_dim
        
        train_set, test_set, train_loader, test_loader = get_data(
            image_size, batch_size=cfg.model.type.batch_size
        )
        
        patcher_module = PatchMakerModule(patch_size, patch_size)
        encoder_module = SequentialFF(
            PatchEncoderFF(num_patches, patch_size, projection_dim, device=device),
            *(
                TransformerBlockFF(num_patches, projection_dim, projection_dim, num_heads)
                for _ in range(transformer_layers)  # N x num_patches x projection_dim
            )
        )
        head_module = th.nn.Sequential(
            th.nn.LayerNorm((num_patches, projection_dim * (transformer_layers - 1)), eps=1e-6), 
            th.nn.Flatten(start_dim=1), 
            th.nn.Dropout(p=0.5), 
            *(
                mlp_block(in_dim, out_dim, 0.0)
                for (in_dim, out_dim) in th.tensor([
                    num_patches * projection_dim * transformer_layers, 2048, 1024
                ]).unfold(0, 2, 1)
            ),
            th.nn.Linear(1024, 10, bias=True), 
            th.nn.Softmax(dim=-1)
        )
        
        optimizer = th.optim.AdamW(
            [
                dict(
                    params=patcher_module.parameters(),
                    lr=cfg.train.lr,
                    weight_decay=cfg.train.lamda,
                ),
                dict(
                    params=encoder_module.parameters(),
                    lr=cfg.train.lr,
                    weight_decay=cfg.train.lamda,
                ),
                dict(
                    params=head_module.parameters(),
                    lr=cfg.train.head_lr,
                    weight_decay=cfg.train.head_lamda,
                )
            ]
        )
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
        
        patcher_module.to(device), encoder_module.to(device), head_module.to(device)
        classification_loss_func = th.nn.MSELoss()
        total_loss = 0
    
        for epoch in trange(epochs):
            block_losses, head_losses = [], []
            for (images, (pos_label, neg_label, neu_label)) in test_loader:
                optimizer.zero_grad()
                images, pos_label = images.to(device), pos_label.to(device)
                neg_label, neu_label = neg_label.to(device), neu_label.to(device)
                
                images = patcher_module(images)
                
                pos_data = th.cat(
                    [images, pos_label[:, None, :].expand(-1, images.size(dim=1), -1)],
                    dim=-1
                )
                neg_data = th.cat(
                    [images, neg_label[:, None, :].expand(-1, images.size(dim=1), -1)],
                    dim=-1
                )
                neu_data = th.cat(
                    [images, neu_label[:, None, :].expand(-1, images.size(dim=1), -1)],
                    dim=-1
                )
                pos_data = th.nn.functional.layer_norm(pos_data, pos_data.size())
                neg_data = th.nn.functional.layer_norm(neg_data, neg_data.size())
                neu_data = th.nn.functional.layer_norm(neu_data, neu_data.size())
                
                data = th.cat([pos_data, neg_data])
                label = th.cat([
                    th.ones((pos_data.size(dim=0), 1), device=device),
                    th.zeros((neg_data.size(dim=0), 1), device=device)
                ])
                
                encoder_module.train()
                encoder_module(data, label)
                block_loss = encoder_module.losses()
                
                encoder_module.eval()
                head_module.eval()
                with th.no_grad():
                    out = encoder_module(neu_data, neu_label)
                
                out = head_module(out.detach())
                head_loss = classification_loss_func(pos_label.float(), out.float())
                loss = th.mean(th.hstack(block_loss + [head_loss]))  
                loss.backward()
                optimizer.step()
                
                block_losses.append(block_loss)
                head_losses.append(head_loss)
                
            scheduler.step()
            block_losses = th.Tensor(block_losses).T.mean(dim=1)
            writer.add_scalars(
                f"{TRAIN_NAME}/transformer_blocks",
                {f"block_{i}": block_losses[i].item() for (i, val) in enumerate(block_losses)},
                epoch
            )
            writer.add_scalar(f"{TRAIN_NAME}/loss_head", th.sum(th.vstack(head_losses)).item(), epoch)
            
            test = True
            if test:
                with th.no_grad():
                    scores = []
                    for (images, (pos_label, neg_label, neu_label)) in test_loader:
                        images, pos_label = images.to(device), pos_label.to(device)
                        neg_label, neu_label = neg_label.to(device), neu_label.to(device)
        
                        images = patcher_module(images)
                        neu_data = th.cat(
                            [images, neu_label[:, None, :].expand(-1, images.size(dim=1), -1)],
                            dim=-1
                        )
                        neu_data = th.nn.functional.layer_norm(neu_data, neu_data.size())
                
                        encoder_module.eval()
                        head_module.eval()
                        with th.no_grad():
                            out = encoder_module(neu_data, neu_label)
                        out = head_module(out.detach())
                
                        prediction = out.argmax(dim=1, keepdims=True)
                        actual = pos_label.argmax(dim=1, keepdims=True)
                
                        accuracy = (prediction == actual).sum() / actual.size(dim=0)
                        scores.append(accuracy)
                        
                    accuracy = th.stack(scores).mean()
                writer.add_scalar(f"{TRAIN_NAME}/accuracy", accuracy, epoch)
                