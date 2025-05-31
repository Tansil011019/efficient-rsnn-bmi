import stork

def get_activation_function(cfg):
    """
    Get the activation function based on the configuration.
    """
    activation_function = stork.activations.CustomSpike
    if cfg.stochastic:
        activation_function.escape_noise_type = "sigmoid"
    else:
        activation_function.escape_noise_type = "step"

    activation_function.escape_noise_params = {"beta": cfg.training.SG_beta}
    activation_function.surrogate_type = "SuperSpike"
    activation_function.surrogate_params = {"beta": cfg.training.SG_beta}

    return activation_function