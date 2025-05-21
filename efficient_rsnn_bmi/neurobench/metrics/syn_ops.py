from neurobench.metrics.workload import SynapticOperations as SynapticOperationsBased
from neurobench.metrics.utils.layers import single_layer_MACs

class SynapticOperations(SynapticOperationsBased):
    def __init__(self):
        super().__init__()

    def __call__(self, model, preds, data):
        for hook in model.connection_hooks:
            inputs = hook.inputs  # copy of the inputs, delete hooks after
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