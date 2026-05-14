import torch
import torch.nn as nn
import numpy as np

class Base(nn.Module):
    def __init__(self, stop_threshold):
        super().__init__()

        self.init_model()
        self.num_params()

        self.register_buffer("step", torch.zeros(1, dtype=torch.long))
        self.register_buffer("stop_threshold", torch.tensor(stop_threshold, dtype=torch.float32))

    @property
    def r(self):
        return self.decoder.r.item()

    @r.setter
    def r(self, value):
        self.decoder.r = self.decoder.r.new_tensor(value, requires_grad=False)

    def init_model(self):
        for p in self.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p)

    def finetune_partial(self, whitelist_layers):
        self.zero_grad()
        for name, child in self.named_children():
            if name in whitelist_layers:
                print("Trainable Layer: %s" % name)
                print("Trainable Parameters: %.3f" % sum([np.prod(p.size()) for p in child.parameters()]))
                for param in child.parameters():
                    param.requires_grad = False

    def get_step(self):
        return self.step.data.item()

    def reset_step(self):
        # assignment to parameters or buffers is overloaded, updates internal dict entry
        self.step = self.step.data.new_tensor(1)

    def log(self, path, msg):
        with open(path, "a") as f:
            print(msg, file=f)

    def load(self, path, device, optimizer=None):
        # Use device of model params as location for loaded state
        checkpoint = torch.load(str(path), map_location=device)

        # determine where the actual state_dict is stored in checkpoint
        if isinstance(checkpoint, dict):
            if "model_state" in checkpoint:
                state = checkpoint["model_state"]
            elif "state_dict" in checkpoint:
                state = checkpoint["state_dict"]
            elif "model" in checkpoint:
                state = checkpoint["model"]
            else:
                # assume checkpoint itself is a state_dict mapping
                state = checkpoint
        else:
            state = checkpoint

        # filter keys to only those that exist in current model and have matching shape
        model_state = self.state_dict()
        filtered_state = {}

        def _strip_module(k):
            return k[7:] if k.startswith('module.') else k

        for k, v in state.items():
            k_stripped = _strip_module(k)
            if k_stripped in model_state:
                if model_state[k_stripped].shape == v.shape:
                    filtered_state[k_stripped] = v
                else:
                    print(f"Skipping parameter {k_stripped}: checkpoint shape={v.shape}, model shape={model_state[k_stripped].shape}")
            else:
                # key not found in current model; skip
                print(f"Skipping parameter {k} as it does not exist in the current model")

        # load filtered parameters
        self.load_state_dict(filtered_state, strict=False)

        # load optimizer state if present
        if isinstance(checkpoint, dict) and "optimizer_state" in checkpoint and optimizer is not None:
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state"])
            except Exception as e:
                print('Could not load optimizer state:', e)

    def save(self, path, optimizer=None):
        if optimizer is not None:
            torch.save({
                "model_state": self.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            }, str(path))
        else:
            torch.save({
                "model_state": self.state_dict(),
            }, str(path))


    def num_params(self, print_out=True):
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
        if print_out:
            print("Trainable Parameters: %.3fM" % parameters)
        return parameters
