from neurobench.metrics.workload import SynapticOperations as SynapticOperationsBased
from neurobench.blocks.layer import STATELESS_LAYERS, RECURRENT_LAYERS, RECURRENT_CELLS
from neurobench.metrics.utils.layers.input import binary_inputs
from neurobench.metrics.utils.layers.macs import stateless_layer_macs, recurrent_cell_macs, recurrent_layer_macs
from neurobench.metrics.utils.layers.binary_copy import binarize_tensor
from DCLS.construct.modules import Dcls1d
import torch.nn.functional as F
from torch.nn.modules.utils import _single
import torch
import copy

def make_binary_copy(layer, all_ones=False):
    """
    Makes a binary copy of the layer.

    Non-zero entries in the layer's weights and biases are set to 1.
    If all_ones is True, all entries (including zeros) are set to 1.

    Args:
        layer (torch.nn.Module): The layer to be binarized.
        all_ones (bool): If True, all entries (including zeros) are set to 1.

    Returns:
        torch.nn.Module: A binary copy of the input layer.

    """
    layer_copy = copy.deepcopy(layer)

    if isinstance(layer, STATELESS_LAYERS):
        layer_copy.weight.data = binarize_tensor(layer_copy.weight.data, all_ones)
        if layer.bias is not None:
            layer_copy.bias.data = binarize_tensor(layer_copy.bias.data, all_ones)

    elif isinstance(layer, RECURRENT_CELLS):
        attribute_names = ["weight_ih", "weight_hh"]
        if layer.bias:
            attribute_names += ["bias_ih", "bias_hh"]

        for attr in attribute_names:
            with torch.no_grad():
                attr_val = getattr(layer_copy, attr)
                setattr(
                    layer_copy,
                    attr,
                    torch.nn.Parameter(binarize_tensor(attr_val.data, all_ones)),
                )

    return layer_copy


def stateless_layer_macs_dcls1d(inputs, layer, total):
    # then multiply the binary layer with the diagonal matrix to get the MACs
    # layer_bin = make_binary_copy(layer, all_ones=total)

    # bias is not considered as a synaptic operation
    # in the future you can change this parameter to include bias
    # bias = False
    # if layer_bin.bias is not None and not bias:
    #     # suppress the bias to zero
    #     layer_bin.bias.data = torch.zeros_like(layer_bin.bias.data)

    # nr_updates = layer_bin(
    #     inputs
    # )  # this returns the number of MACs for every output neuron: if spiking neurons only AC

    layer_copy = copy.deepcopy(layer)
    weight = layer_copy.weight
    P = layer_copy.P
    SIG = layer_copy.SIG

    kernel_weight = layer_copy.DCK(weight, P, SIG)
    # print(f"Kernel weight: {kernel_weight}")
    # print(f"Kernel weight shape: {kernel_weight.shape}")

    binary_kernel_weight = binarize_tensor(kernel_weight, all_ones=total)
    if layer.bias is not None:
        layer_copy.bias = binarize_tensor(layer_copy.bias, all_ones=total)
    
    # print(f"Binary Kernel weight: {binary_kernel_weight}")
    # print(f"Binary Kernel weight shape: {binary_kernel_weight.shape}")

    # print(f"Inputs: {inputs}")
    
    conv_out = F.conv1d(
        inputs,
        binary_kernel_weight.to(inputs.dtype),
        layer_copy.bias,
        layer_copy.stride,
        _single(0),
        _single(1),
        layer_copy.groups
    )

    # print(f"Convolutional output: {conv_out}")
    # print(f"Convolutional output shape: {conv_out.shape}")
    # print(f"If there is negative error: {(conv_out < 0).any()}")

    # print(conv_out.sum())

    # raise ValueError("Breakpoint")


    # return

    # if (nr_updates < 0).any():
    #     print(f"nr updates: {nr_updates}")
    #     print(f"nr updates shape: {nr_updates.shape}")
    #     print(f"Sum: {nr_updates.sum()}")
    return conv_out.to(torch.float32).sum()

def single_layer_MACs(inputs, layer, total=False):
    """
    Computes the MACs for a single layer.

    returns effective operations if total=False, else total operations (including zero operations)
    Supported layers: Linear, Conv1d, Conv2d, Conv3d, RNNCellBase, LSTMCell, GRUCell

    """
    macs = 0

    # copy input
    # print(f"Input: {inputs[0][0]}")
    # print(f"Input Shape: {inputs.shape}")
    inputs, spiking, in_states = binary_inputs(inputs, all_ones=total)

    if isinstance(layer, STATELESS_LAYERS):
        macs = stateless_layer_macs(inputs, layer, total)
    elif isinstance(layer, RECURRENT_LAYERS):
        macs = recurrent_layer_macs(inputs, layer, total)
    elif isinstance(layer, RECURRENT_CELLS):
        macs = recurrent_cell_macs(inputs, layer, total, in_states)
    elif isinstance(layer, Dcls1d):
        macs = stateless_layer_macs_dcls1d(inputs, layer, total)
    
    # print(f"Macs: {int(macs)}")
    
    # raise ValueError('Breakpoint')
    # print(f"MACs: {macs}")

    return int(macs), spiking

class SynapticOperations(SynapticOperationsBased):
    def __init__(self):
        super().__init__()

    def __call__(self, model, preds, data):
        for hook in model.connection_hooks:
            # print(hook)
            # print(hook.layer)
            inputs = hook.inputs  # copy of the inputs, delete hooks after
            # print(len(inputs))
            for spikes in inputs:
                # spikes is batch, features, see snntorchmodel wrappper
                # for single_in in spikes:
                if len(spikes) == 1:
                    spikes = spikes[0]
                hook.hook.remove()
                operations, spiking = single_layer_MACs(spikes, hook.layer)
                total_ops, _ = single_layer_MACs(spikes, hook.layer, total=True)
                self.total_synops += total_ops
                if spiking:
                    self.AC += operations
                else:
                    self.MAC += operations
                hook.register_hook()
        # ops_per_sample = ops / data[0].size(0)
        self.total_samples += data[0].squeeze().size(0)
        return self.compute()