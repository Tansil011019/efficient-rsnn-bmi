from stork.nodes.input import InputGroup

import torch

class CustomInputGroup(InputGroup):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def feed_data(self, data, padding=None):
        '''
        padding is for additional input for the rest of the delay
        '''
        reshaped = data.reshape((data.shape[:2] + self.shape)).to(self.device)

        if padding is not None:
            pad = torch.zeros((reshaped.shape[0], padding, reshaped.shape[2]), device=self.device)
            reshaped = torch.cat([reshaped, pad], dim=1)
        
        self.local_data = reshaped
    
    def get_flattened_out_sequence(self):
        return self.get_state_sequence("out")