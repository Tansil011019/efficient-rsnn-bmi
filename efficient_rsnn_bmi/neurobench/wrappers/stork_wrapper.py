from neurobench.models.torch_model import TorchModel
from neurobench.hooks.neuron import NeuronHook
from neurobench.hooks.layer import LayerHook

class StorkModel(TorchModel):
    def __init__(self, net):
        super().__init__(net)

    def __call__(self, batch):
        preds_label = self.net.predict(batch).detach().cpu()
        return preds_label
    
    def activation_layers(self):
        return self.net.groups[1:-1]
    
    def register_hooks(self):
        self.cleanup_hooks()

        for layer in self.activation_layers():
            layer_name = layer.name
            self.activation_hooks.append(NeuronHook(layer, layer_name))
        
        for layer in self.connection_layers():
            self.connection_hooks.append(LayerHook(layer))