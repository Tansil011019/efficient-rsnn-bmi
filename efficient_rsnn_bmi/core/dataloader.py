import torch
from torch.utils.data import Dataset

from data.config.dataloader import DatasetLoaderConfig
from data.neurobench.dataloader import DatasetLoader

from efficient_rsnn_bmi.utils.logger import get_logger

logger = get_logger(__name__)

def get_dataloader(cfg, dtype=torch.float32):

    dataloader = DatasetLoader(
        basepath=cfg.datasets.data_dir,
        ratio_val=cfg.datasets.ratio_val,
        random_val=cfg.datasets.random_val,
        extend_data=cfg.datasets.extend_data,
        sample_duration=cfg.datasets.sample_duration,
        remove_segments_inactive=cfg.datasets.remove_segments_inactive,
        p_drop=cfg.datasets.p_drop,
        p_insert=cfg.datasets.p_insert,
        jitter_sigma=cfg.datasets.jitter_sigma,
        dtype=dtype
    )

    return dataloader

def compute_input_firing_rates(data, cfg):
    mean1 = 0
    mean2 = 0

    for i in range(len(data)):
        mean1 += torch.sum(data[i][0][:, :96]) / cfg.datasets.sample_duration / 96
        try:
            mean2 += torch.sum(data[i][0][:, 96:]) / cfg.datasets.sample_duration / 96
        except:
            continue

    mean1 /= len(data)
    mean2 /= len(data)

    # For LOCO
    if data[0][0].shape[1] == 192:
        return mean1, mean2

    # FOR INDY
    else:
        return mean1, None