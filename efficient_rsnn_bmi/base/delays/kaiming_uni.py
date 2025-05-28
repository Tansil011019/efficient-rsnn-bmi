from stork.initializers import Initializer
from torch.nn import init
import json

class KaimingUniformInitializer(Initializer):
    def __init__(
            self,
            sig_init,
            a, b,
            nonlinearity='relu', 
            sparsity_p=0.0,
            verbose=False,
            **kwargs
        ):
        super().__init__(**kwargs)
        self.nonlinearity = nonlinearity
        self.sig_init = sig_init
        self.init_pos_a = a 
        self.init_pos_b = b 
        self.sparsity_p = sparsity_p
        self.verbose = verbose

    def _get_weights(self, connection):
        weights = init.kaiming_uniform_(connection.op.weight, nonlinearity=self.nonlinearity)
        return weights
    
    def _set_position(self, connection):
        init.uniform_(connection.op.P, a=self.init_pos_a, b=self.init_pos_b)
        connection.op.clamp_parameters()
    
    def _set_sig(self, connection):
        init.constant_(connection.op.SIG, self.sig_init)
        connection.op.SIG.requires_grad = False


    def initialize_connection(self, connection):
        weights = self._get_weights(connection)
        if (self.verbose):
            weight_before = connection.op.weight.clone()
            position_before = connection.op.P.clone()
            sig_before = connection.op.SIG.clone()


        self._set_weights_and_bias(connection, weights)
        self._set_position(connection)
        self._set_sig(connection)
        if (self.verbose):
            weight_after = connection.op.weight
            position_after = connection.op.P
            sig_after = connection.op.SIG
            print("=" * 50)
            print("WEIGHT")
            print("=" * 50)
            print(f"Before - \nmean: {weight_before.mean():.4f}, std: {weight_before.std():.4f}")
            print(f"After  - \nmean: {weight_after.mean():.4f}, std: {weight_after.std():.4f}")
            print("=" * 50)
            print("POSITION")
            print("=" * 50)
            print(f"Before - \nmean: {position_before.mean():.4f}, std: {position_before.std():.4f}")
            print(f"After  - \nmean: {position_after.mean():.4f}, std: {position_after.std():.4f}")
            print("=" * 50)
            print("SIG")
            print("=" * 50)
            print(f"Before - \nmean: {sig_before.mean():.4f}, std: {sig_before.std():.4f}")
            print(f"After  - \nmean: {sig_after.mean():.4f}, std: {sig_after.std():.4f}")

            weight = {
                "before": weight_before.detach().cpu().flatten().tolist(),
                "after": weight_after.detach().cpu().flatten().tolist()
            }
            position = {
                "before": position_before.detach().cpu().flatten().tolist(),
                "after": position_after.detach().cpu().flatten().tolist()
            }
            sig = {
                "before": sig_before.detach().cpu().flatten().tolist(),
                "after": sig_after.detach().cpu().flatten().tolist()
            }

            with open('init_delay_weight.json', "w") as f:
                json.dump(weight, f)
            with open('init_delay_position.json', "w") as f:
                json.dump(position, f)    
            with open('init_delay_sig.json', "w") as f:
                json.dump(sig, f)
    