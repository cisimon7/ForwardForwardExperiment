import torch as th
from cifar10.dataset import get_data
from torch.utils.data import Subset
from utils import plot3ch, confusion_map
import sys
import torch as th
from utils import seed_devices
from cifar10.dataset import get_data
from hydra import compose, initialize
from layers import get_model, SequentialFF, LinearLayerFF


if __name__ == "__main__":
    with initialize(version_base=None, config_path="../../conf"):
        cfg = compose(config_name="config", overrides=["model=cifar10", "train=cifar10", "model/type=fcn"])
        seed_devices(cfg.seed)
    
        train_set, test_set, train_loader, test_loader = get_data(
            32, cfg.model.type.batch_size, cfg.model.type.num_workers
        )
        model_ff = SequentialFF(
            LinearLayerFF(
                (3*32*32) + 10, 1_000,
                cfg.model.type.theta,
                ratio_peer=cfg.model.type.peer_ratio,
            ),
            *(
                LinearLayerFF(
                    1_000, 1_000,
                    cfg.model.type.theta,
                    ratio_peer=cfg.model.type.peer_ratio,
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
        )
        
        model_ff.load_state_dict(th.load(f"models/cifar10/fcn/model_ff_best"))
        head.load_state_dict(th.load(f"models/cifar10/fcn/head_best"))
        
        # sample = Subset(test_set, th.randint(low=0, high=len(test_set), size=(64,)))
        sample = test_set
        images = th.stack([image for (image, _) in sample])
        neu_labels = th.stack([neu_label for (_, (_, _, neu_label)) in sample])
        pos_labels = th.stack([pos_label for (_, (pos_label, _, _)) in sample])
    
        neu_data = th.cat([images.flatten(start_dim=1), neu_labels], dim=1)
        neu_data = th.nn.functional.layer_norm(neu_data, neu_data.size())

        model_ff.eval()
        head.eval()
        with th.no_grad():
            out = model_ff(neu_data, neu_labels)
        out = head(out)

        predictions = out.argmax(dim=1, keepdims=True)
        actuals = pos_labels.argmax(dim=1, keepdims=True)
        class_names = train_set.class_names
        
        # plot3ch(images, predictions.flatten().tolist(), class_names)
        confusion_map(actuals.flatten(), predictions.flatten(), 10)
