from stork.nodes.input import InputGroup

class InterpolationInputGroup(InputGroup):
    def __init__(self, *args, **kwargs):
        # print("=" * 50)
        # print("INTERPOLATION INPUT GROUP INIT")
        super().__init__(*args, **kwargs)
        # print("=" * 50)

    def forward(self):
        # print("=" * 50)
        # print("INPUT GROUP FORWARD FOR INTERPOLATION")
        # print(f"Local Data Shape: {self.local_data.shape}") # [250, 500, 96]
        self.out = self.states["out"] = self.local_data[:, self.clk] # [250, 96])
        # print(f"Out Shape From Input Group Forward: {self.out.shape}")
        # print("IGFI=" * 50)