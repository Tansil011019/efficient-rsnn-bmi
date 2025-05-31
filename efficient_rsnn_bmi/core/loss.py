import torch
from efficient_rsnn_bmi.base.loss import MeanSquareError, RootMeanSquareError, MeanAbsoluteError, HuberLoss

def _choose_loss(cfg):

    if cfg.training.loss == "MSE":
        loss_class = MeanSquareError
    elif cfg.training.loss == "RMSE":
        loss_class = RootMeanSquareError
    elif cfg.training.loss == "MAE":
        loss_class = MeanAbsoluteError
    elif cfg.training.loss == "Huber":
        loss_class = HuberLoss
    else:
        raise ValueError(f"Unknown loss: {cfg.training.loss}")

    return loss_class

def get_train_loss(cfg, nb_time_steps):

    loss_class = _choose_loss(cfg)
    
    args = {}  
    
    # Mask early timesteps
    if cfg.training.mask_early_timesteps:
        mask = torch.ones(nb_time_steps)
        mask[: cfg.training.nb_masked_timesteps] = 0
        mask = torch.stack([mask, mask], dim=1)
        
        args["mask"] = mask

    return loss_class(**args)