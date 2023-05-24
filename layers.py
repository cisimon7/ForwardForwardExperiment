import sys
import math
import torch as th
from utils import layer_norm
from omegaconf import DictConfig


class LinearLayerFF(th.nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            threshold: int = 0, ratio_peer=0,
            bias: bool = True, act_func=th.nn.ReLU(inplace=True),
            loss_func=th.nn.BCEWithLogitsLoss(),
            device=None, dtype=None
    ):
        super(LinearLayerFF, self).__init__()
        
        self.accuracy, self.idx = 0, 0
        self.loss, self.peer_loss = None, None
        self.threshold, self.ratio_peer = threshold, ratio_peer
        self.linear = th.nn.Linear(in_features, out_features, bias, device, dtype)
        self.loss_func, self.act_func = loss_func, act_func
        self.peer_mean = th.zeros(out_features, device=device) + 0.5
        self.setup()
        
    def setup(self):
        th.nn.init.normal_(
            self.linear.weight,
            mean=0, std=1/math.sqrt(self.linear.weight.shape[0])
        )
        th.nn.init.zeros_(self.linear.bias)
        
    def set_loss_accuracy(self, z, labels):
        # logits = th.sum(z**2, dim=-1, keepdim=True) - self.threshold
        logits = th.sum(z**2, dim=-1, keepdim=True) - z.shape[1]
        self.loss = self.loss_func(logits, labels)  # Equation 1
        
        with th.no_grad():
            self.accuracy = (th.sum((logits > 0.5) == labels) / z.size(dim=0)).item()
        
        return self.loss, self.accuracy
    
    def set_peer_normalization_loss(self, x, positive_idx):
        """
        Prevents any of the hidden units from being extremely active or permanently off
        """
        gamma = 0.9
        peer_mean = th.mean(x[positive_idx], dim=0)
        self.peer_mean = (gamma * self.peer_mean.detach()) + (peer_mean * (1 - gamma))
        
        self.peer_loss = th.mean((th.mean(self.peer_mean) - self.peer_mean) ** 2)
        return self.peer_loss
        
    def forward(self, x, labels):
        x = self.linear(x)
        x = self.act_func(x)
        
        if self.training:
            self.set_loss_accuracy(x, labels)
            peer_loss = self.set_peer_normalization_loss(x, th.nonzero(labels == 1))
            self.loss += (peer_loss * self.ratio_peer)
            
        x = x.detach()
        x = th.nn.functional.layer_norm(x, x.size())
        # x = layer_norm(x)
        return x
    
    
class SequentialFF(th.nn.Module):
    def __init__(self, *args: LinearLayerFF):
        super(SequentialFF, self).__init__()
        
        self.model = th.nn.ModuleList([args[0]])
        for layer in args[1:]:
            self.model.append(layer)
    
    def forward(self, x, labels):
        return th.cat([x := layer(x, labels) for layer in self.model][1:], dim=-1)
    
    def losses(self):
        return [layer.loss for layer in self.model]
    
    
def get_model(cfg: DictConfig):
    model_ff = SequentialFF(
        LinearLayerFF(
            (28*28) + 10, 1_000,
            cfg.model.theta,
            ratio_peer=cfg.model.peer_ratio,
            device=cfg.device
        ),
        *(
            LinearLayerFF(
                1_000, 1_000,
                cfg.model.theta,
                ratio_peer=cfg.model.peer_ratio,
                device=cfg.device
            )
            for _ in range(1, cfg.model.num_layers)
        )
    )
    head = th.nn.Sequential(
        th.nn.Linear((cfg.model.num_layers - 1) * 1_000, 10, bias=False),
        th.nn.Softmax(dim=1)
    )
    th.nn.init.zeros_(head[0].weight)
    
    return model_ff, head
