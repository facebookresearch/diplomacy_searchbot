import torch

from .train_sl import new_model


def load_dipnet_model(checkpoint_path, map_location="cpu", eval=False):
    checkpoint = torch.load(checkpoint_path, map_location=map_location)

    model = new_model(checkpoint["args"])

    # strip "module." prefix if model was saved with DistributedDataParallel wrapper
    state_dict = {
        (k[len("module.") :] if k.startswith("module.") else k): v
        for k, v in checkpoint["model"].items()
    }
    model.load_state_dict(state_dict)

    if eval:
        model.eval()

    return model
