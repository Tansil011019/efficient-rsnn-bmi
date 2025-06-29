from neurobench.models.torch_model import TorchModel
from neurobench.hooks.neuron import NeuronHook
from neurobench.hooks.layer import LayerHook
from neurobench.blocks.layer import SUPPORTED_LAYERS
from DCLS.construct.modules import Dcls1d

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

    def connection_layers(self):
        """
        Retrieve all connection layers in the network.

        Connection layers include Linear, Conv, and RNN-based layers.

        Returns:
            list: Connection layers.

        """

        def find_connection_layers(module):
            """Recursively find connection layers in a module."""
            layers = []
            for child in module.children():
                if isinstance(child, SUPPORTED_LAYERS + (Dcls1d, )):
                    layers.append(child)
                elif list(child.children()):  # Check for nested submodules
                    layers.extend(find_connection_layers(child))
            return layers

        return find_connection_layers(self.__net__())