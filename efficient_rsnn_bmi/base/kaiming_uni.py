from stork.initializers import Initializer
import torch
from torch.nn import init

import math
import json

class KaimingUniformInitializer(Initializer):
    def __init__(
            self,
            sig_init,
            a, b,
            nonlinearity='relu', 
            sparsity_p=0.0,
            **kwargs
        ):
        super().__init__(**kwargs)
        self.nonlinearity = nonlinearity
        self.sig_init = sig_init
        self.init_pos_a = a 
        self.init_pos_b = b 
        self.sparsity_p=sparsity_p

    def _get_weights(self, connection):

        weights = init.kaiming_uniform_(connection.op.weight, nonlinearity=self.nonlinearity)

        # TODO: I skipped this first for now
        if self.sparsity_p > 0:
            with torch.no_grad():
                pass
        return weights
    
    def _set_position(self, connection):
        init.uniform_(connection.op.P, a=self.init_pos_a, b=self.init_pos_b)
        connection.op.clamp_parameters()
    
    def _set_sig(self, connection):
        init.constant_(connection.op.SIG, self.sig_init)
        connection.op.SIG.requires_grad = False


    def initialize_connection(self, connection, verbose=False):
        # before = connection.op.weight.clone()
        weights = self._get_weights(connection)
        self._set_weights_and_bias(connection, weights)
        self._set_position(connection)
        self._set_sig(connection)

        

        # after = connection.op.weight

        # if True: # dont forget to remove this
        #     weight = {
        #         "before": before.detach().cpu().flatten().tolist(),
        #         "after": after.detach().cpu().flatten().tolist()
        #     }
        #     with open('init_delay_weight.json', "w") as f:
        #         json.dump(weight, f)
            
        #     print(f"Before - mean: {before.mean():.4f}, std: {before.std():.4f}")
        #     print(f"After  - mean: {after.mean():.4f}, std: {after.std():.4f}")
    