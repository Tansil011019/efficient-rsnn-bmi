from typing import TypeVar, Type
from omegaconf import OmegaConf, DictConfig

import gc
import torch

T = TypeVar("T")

def from_config(cfg: DictConfig, schema: Type[T]) -> T:
    """
    Convert a DictConfig to a dataclass instance.

    Args:
        cfg (DictConfig): The configuration to convert.
        schema (Type[T]): The dataclass type to convert to.

    Returns:
        T: An instance of the dataclass with the configuration values.
    """
    raw = OmegaConf.to_container(cfg, resolve=True)
    filtered = {k: raw[k] for k in schema.__annotations__ if k in raw}
    return schema(**filtered)

def clean_gpu():
    gc.collect()
    torch.cuda.empty_cache()