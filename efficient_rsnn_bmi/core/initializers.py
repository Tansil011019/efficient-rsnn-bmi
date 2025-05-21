import torch
from stork.initializers import (
    FluctuationDrivenCenteredNormalInitializer,
    DistInitializer,
)

from efficient_rsnn_bmi.utils.logger import get_logger

logger = get_logger(__name__)

def get_initializers(cfg, nu=None, dtype=torch.float32):
    """
    Get the initializers based on the configuration.
    """
    if nu is None or cfg.initializer.compute_nu is False:
        nu = cfg.initializer.nu
    else:
        logger.info(f"Initializing with nu = {nu}")

    hidden_init = FluctuationDrivenCenteredNormalInitializer(
        sigma_u=cfg.initializer.sigma_u,
        nu=nu,
        timestep=cfg.datasets.dt,
        alpha=cfg.initializer.alpha,
        dtype=dtype
    )

    readout_init = DistInitializer(
        dist=torch.distributions.Normal(0, 1), 
        scaling="1/sqrt(k)", 
        dtype=dtype
    )

    return hidden_init, readout_init