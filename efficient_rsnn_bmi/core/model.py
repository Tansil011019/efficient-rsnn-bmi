import torch

from efficient_rsnn_bmi.experiments.models.rsnn.rsnn import BaselineRecurrentSpikingModel
from efficient_rsnn_bmi.base.lif import CustomLIFGroup
from efficient_rsnn_bmi.base.readout import CustomReadoutGroup, AverageReadouts

from stork.nodes import InputGroup
from stork.layers import Layer
from stork.connections import Connection

from .activation import get_activation_function
from .regularization import get_regularizers
from .dataloader import compute_input_firing_rates
from .initializers import get_initializers
from .readout import get_custom_readouts

from efficient_rsnn_bmi.utils.logger import get_logger

logger = get_logger(__name__)

def get_model (cfg, nb_inputs, dtype, data=None):
    """
    Get the model based on the configuration.
    """
    nb_time_steps = int(cfg.datasets.sample_duration / cfg.datasets.dt)
    nb_outputs = cfg.datasets.nb_outputs
    
    model = BaselineRecurrentSpikingModel(
        cfg.training.batchsize,
        nb_time_steps = nb_time_steps,
        nb_inputs = nb_inputs,
        device = cfg.device,
        dtype = dtype,
    )

    activation_function = get_activation_function(cfg)

    regularizers = get_regularizers(cfg)

    if data is not None:
        mean1, mean2 = compute_input_firing_rates(data, cfg)
    else:
        mean1 = None

    hidden_init, readout_init = get_initializers(cfg, mean1, dtype)

    input_group = model.add_group(
        InputGroup(
            nb_inputs,
            dropout_p = cfg.model.dropout_p,
        )
    )

    current_src_grp = input_group

    hidden_neuron_kwargs = {
        "tau_mem": cfg.model.tau_mem,
        "tau_syn": cfg.model.tau_syn,
        "activation": activation_function,
        "dropout_p": cfg.model.dropout_p,
        "het_timescales": cfg.model.het_timescales,
        "learn_timescales": cfg.model.learn_timescales,
        "is_delta_syn": cfg.model.delta_synapses,
    }

    for i in range(cfg.model.nb_hidden):
        hidden_layer = Layer(
            name='hidden',
            model = model,
            size= cfg.model.hidden_size[i],
            input_group = current_src_grp,
            recurrent = cfg.model.recurrent[i],
            regs= regularizers,
            neuron_class=CustomLIFGroup,
            neuron_kwargs=hidden_neuron_kwargs,
            connection_kwargs={}
        )

        current_src_grp = hidden_layer.output_group

        hidden_init.initialize(hidden_layer)

        if i == 0 and nb_inputs == 192 and data is not None:
            with torch.no_grad():
                hidden_layer.connections[0].get_weights()[:, :96] /= mean2 / mean1

        if cfg.model.multiple_readouts:
            logger.info("Adding custom readout groups")
            custom_readouts = get_custom_readouts(cfg)
            for g in custom_readouts:
                model.add_group(g)
                con_ro = model.add_connection(Connection(current_src_grp, g, dtype=dtype))
                readout_init.initialize(con_ro)
            
            model.add_group(AverageReadouts(model.groups[-len(custom_readouts) :]))
        
        else:
            logger.info("Adding single readout group")
            readout_group = model.add_group(
                CustomReadoutGroup(
                    nb_outputs,
                    tau_mem=cfg.model.tau_mem_readout,
                    tau_syn=cfg.model.tau_syn_readout,
                    het_timescales=cfg.model.het_timescales_readout,
                    learn_timescales=cfg.model.learn_timescales_readout,
                    initial_state=-1e-2,
                    is_delta_syn=cfg.model.delta_synapses,
                )
            )

            print("Readout group shape: ", readout_group.shape)
            print("Current source group shape: ", current_src_grp.shape)
            print(current_src_grp, readout_group)
            con_ro = model.add_connection(
                Connection(current_src_grp, readout_group, dtype=dtype)
            )

            readout_init.initialize(con_ro)

    return model