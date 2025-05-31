import torch
import stork

def get_optimizer(cfg, dtype):
    
    opt_kwargs = {
        "lr": cfg.training.lr,
        **({"lr_P": cfg.training.lr_P} if cfg.name == 'synaps-delay' else {})
    }
    
    if cfg.training.optimizer == "adam":
        opt = torch.optim.Adam
        opt_kwargs["eps"] = 1e-4 if dtype == torch.float16 else 1e-8
        
    elif cfg.training.optimizer == "SMORMS3":
        opt = stork.optimizers.SMORMS3
        opt_kwargs["eps"] = 1e-5 if dtype == torch.float16 else 1e-16
        
    return opt, opt_kwargs