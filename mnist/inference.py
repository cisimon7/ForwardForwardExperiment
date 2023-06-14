import hydra
import torch as th
from dataset import get_data
from layers import get_model
from omegaconf import DictConfig
from torch.utils.data import Subset
from utils import plot, confusion_map


TRAIN_NAME = "train_00"


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def inference(cfg: DictConfig):
    train_set, test_set, train_loader, test_loader = get_data(
        cfg.model.batch_size, cfg.model.num_workers
    )
    model_ff, head = get_model(cfg)
    
    model_ff.load_state_dict(th.load(f"models/{TRAIN_NAME}/model_ff"))
    head.load_state_dict(th.load(f"models/{TRAIN_NAME}/head"))
    model_ff.eval()
    
    sample = Subset(test_set, th.randint(low=0, high=len(test_set), size=(64,)))
    images = th.stack([image for (image, _) in sample])
    pos_labels = th.stack([pos_label for (_, (pos_label, _, _)) in sample])
    
    neu_data = th.cat([images.flatten(start_dim=1), pos_labels], dim=1)
    neu_data = th.nn.functional.layer_norm(neu_data, neu_data.size())
    
    with th.no_grad():
        out = model_ff(neu_data, pos_labels)
    out = head(out.detach())
    
    predictions = out.argmax(dim=1, keepdims=True)
    actuals = pos_labels.argmax(dim=1, keepdims=True)
    
    plot(images, predictions.flatten().tolist())
    # confusion_map(actuals.flatten(), predictions.flatten(), 10)


if __name__ == "__main__":
    inference()
