import torch
from hydra.utils import instantiate, get_class

from stork.layers import Layer

from efficient_rsnn_bmi.base.readout import AverageReadouts

from .activation import get_activation_function
from .regularization import get_regularizers
from .initializers import get_initializers
from .readout import get_custom_readouts

from efficient_rsnn_bmi.utils.logger import get_logger

logger = get_logger(__name__)

def get_model (
    config, # just experiment config
    nb_inputs, 
    nb_outputs,
    dt,
    input_firing_rates,
    nb_time_steps,
    dtype, 
    device='cpu',
    max_delay = None,
    verbose=False,
):
    model = instantiate(
        config.model,
        batch_size=config.training.batchsize,
        nb_time_steps=nb_time_steps,
        nb_inputs = nb_inputs,
        device=device,
        dtype=dtype
    )

    activation_function = get_activation_function(config)
    regularizers = get_regularizers(config)

    mean1, mean2 = input_firing_rates

    hidden_init, readout_init = get_initializers(
        config, 
        dt=dt, 
        nu=mean1,
        max_delay=max_delay,
        dtype=dtype,
    )

    input_group = model.add_group(
        instantiate(
            config.input,
            shape=nb_inputs,
        )
    )

    current_src_grp = input_group

    neuron_class = get_class(config.neuron._target_)
    neuron_kwargs = {k: v for k, v in config.neuron.items() if k != '_target_'}
    neuron_kwargs = {
        **neuron_kwargs,
        "activation": activation_function
    }
    
    if config.name == "synaps-delay" and max_delay is not None:
        connection_class = get_class(config.connection_delay._target_)
        connection_kwargs = {k: v for k, v in config.connection_delay.items() if k != '_target_'}
        connection_kwargs = {
            **connection_kwargs,
            "dilated_kernel_size": max_delay,
            "left_padding": max_delay - 1,
            "right_padding": (max_delay - 1) // 2,
            "sig_init": max_delay // 2,
        }
    else:
        connection_class = get_class(config.connection._target_)
        connection_kwargs = {k: v for k, v in config.connection.items() if k != '_target_'}

    for i in range(config.nb_hidden):
        hidden_layer = Layer(
            name='hidden',
            model=model,
            size=config.hidden_size[i],
            input_group=current_src_grp,
            recurrent=config.recurrent[i],
            regs=regularizers,
            neuron_class=neuron_class,
            neuron_kwargs=neuron_kwargs,
            connection_class=connection_class,
            connection_kwargs=connection_kwargs
        )

        current_src_grp = hidden_layer.output_group
        hidden_init.initialize(hidden_layer)

        if i == 0 and nb_inputs == 192 and mean1 is not None:
            with torch.no_grad():
                hidden_layer.connections[0].get_weights()[:, :96] /= mean2 / mean1

        if config.multiple_readouts:
            # custom_readouts = get_custom_readouts(config)
            # custom_readouts = get_custom_readouts(cfg)
            # for g in custom_readouts:
            #     model.add_group(g)
            #     con_ro = model.add_connection(Connection(current_src_grp, g, dtype=dtype))
            #     readout_init.initialize(con_ro)
            
            # model.add_group(AverageReadouts(model.groups[-len(custom_readouts) :]))
            raise NotImplementedError("Multiple Readout Groups Haven't Implemented Yet")
        else: 
            readout_group = model.add_group(
                instantiate(
                    config.readout,
                    shape=nb_outputs,
                )
            )
            conn_ro = model.add_connection(
                instantiate(
                    config.connection,
                    src=current_src_grp,
                    dst=readout_group,
                    dtype=dtype,
                )
            )
            readout_init.initialize(conn_ro)
    
    if verbose:
        logger.info("=" * 50)
        logger.info(f"Summary:{model.summary()}")
        logger.info("=" * 50)
    return model