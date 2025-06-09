from hydra.utils import instantiate

import torch
from stork.initializers import (
    DistInitializer,
)

from efficient_rsnn_bmi.utils.logger import get_logger

logger = get_logger(__name__)

def get_initializers(
        config, 
        dt,
        nu=None, 
        max_delay=None,
        dtype=torch.float32
    ):
    if config.initializer_name == 'kaiming' and max_delay is not None:
        hidden_init = instantiate(
            config.initializer,
            sig_init=max_delay // 2,
            a=-max_delay // 2,
            b=max_delay//2,
        )
        logger.info("Kaiming Initializer")

    else: 
        if nu is None or config.compute_nu is False:
            nu = config.initializer.nu
        else:
            logger.info(f"Initializing with nu = {nu}")

        hidden_init = instantiate(
            config.initializer,
            nu=nu,
            dtype=dtype,
            timestep=dt
        )
        logger.info("Fluctuation Initializer")

    readout_init = DistInitializer(
        dist=torch.distributions.Normal(0, 1),
        scaling="1/sqrt(k)", 
        dtype=dtype
    )

    return hidden_init, readout_init