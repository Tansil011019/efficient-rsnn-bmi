from stork.nodes.input import InputGroup

class InterpolationInputGroup(InputGroup):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def get_flattened_out_sequence(self):
        flat_shape = self.get_state_sequence("out").shape
        return self.get_state_sequence("out").reshape(flat_shape)  