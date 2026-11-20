from dataclasses import asdict, dataclass

import yaml


@dataclass
class Config:
    name: str = "baseline"
    seed: int = 0

    image_size: int = 16
    batch_size: int = 128

    T: int = 1000
    schedule: str = "linear"  # linear | cosine
    beta_start: float = 1e-4
    beta_end: float = 0.02

    model: str = "unet"  # unet | unet_noskip | unet_not | resunet
    base_ch: int = 32
    emb_dim: int = 128

    epochs: int = 20
    lr: float = 2e-4

    sample_every: int = 5
    n_samples: int = 64
    checkpoint_every: int = 0
    power_w: float = 0.0  # 0 = device default value

    def save(self, path):
        with open(path, "w") as f:
            yaml.safe_dump(asdict(self), f, sort_keys=False)


def load_config(path=None, overrides=()):
    values = {}
    if path:
        with open(path) as f:
            values.update(yaml.safe_load(f) or {})
    for item in overrides:
        key, val = item.split("=", 1)
        values[key] = yaml.safe_load(val)

    cfg = Config()
    for key, val in values.items():
        if not hasattr(cfg, key):
            raise KeyError(f"Unknown hyperparameter: {key}")
        default = getattr(cfg, key)
        # YAML reads "1e-3" as a string.
        if isinstance(default, (int, float)) and not isinstance(default, bool):
            val = type(default)(float(val))
        setattr(cfg, key, val)
    return cfg
