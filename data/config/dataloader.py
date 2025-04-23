import torch

from dataclasses import dataclass
from typing import Optional, Tuple, Literal

@dataclass
class DatasetLoaderConfig:
    """Configuration for the DataLoader."""
    file_path: str
    filename: str
    val_ratio: float = 0.1
    num_steps: int = 1
    train_ratio: float = 0.8
    label_series: bool = False
    biological_delay: int = 0
    spike_sorting: bool = False
    stride: float = 0.004
    bin_width: float = 0.028
    max_segment_length: int = 2000
    split_num: int = 1
    remove_segments_inactive: bool = False
    download: bool = True
    random_val: bool = False
    extend_data: bool = True
    sample_duration: int = 2
    dtype: torch.dtype = torch.float32
    p_drop: float = 0.0
    p_insert: float = 0.0
    jitter_sigma: float = 0.0