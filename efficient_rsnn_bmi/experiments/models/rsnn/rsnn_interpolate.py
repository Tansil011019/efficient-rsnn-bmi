import torch
import stork
from stork.models import (
    loss_stacks,
    generators
)

from efficient_rsnn_bmi.base.interpolation.lif import InterpolationLIFGroup

from .rsnn import BaselineRecurrentSpikingModel

class InterpolateRecurrentSpikingModel(BaselineRecurrentSpikingModel):
    '''
    The interpolation here just using the linear interpolation
    '''
    def __init__(self, n_keys=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert n_keys >= 1, "n_keys must be greater than or equal to 1"
        self.n_keys = n_keys

    def configure(
        self,
        input,
        output,
        loss_stack=None,
        optimizer=None,
        optimizer_kwargs=None,
        scheduler=None,
        scheduler_kwargs=None,
        generator=None,
        time_step=1e-3,
        wandb=None,
    ):
        self.input_group = input
        self.output_group = output
        self.time_step = time_step
        self.wandb = wandb

        if loss_stack is not None:
            self.loss_stack = loss_stack
        else:
            self.loss_stack = loss_stacks.TemporalCrossEntropyReadoutStack()

        if generator is None:
            self.data_generator_ = generators.StandardGenerator()
        else:
            self.data_generator_ = generator

        # configure data generator
        self.data_generator_.configure(
            self.batch_size,
            self.nb_time_steps,
            self.nb_inputs,
            self.time_step,
            device=self.device,
            dtype=self.dtype,
        )

        for o in self.groups + self.connections:
            o.configure(
                self.batch_size,
                self.nb_time_steps,
                self.time_step,
                self.device,
                self.dtype,
            )

        if optimizer is None:
            optimizer = torch.optim.Adam

        if optimizer_kwargs is None:
            optimizer_kwargs = dict(lr=1e-3, betas=(0.9, 0.999))

        self.optimizer_class = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.configure_optimizer(self.optimizer_class, self.optimizer_kwargs)
        
        self.scheduler_class = scheduler
        self.scheduler_kwargs = scheduler_kwargs
        self.configure_scheduler(self.scheduler_class, self.scheduler_kwargs)
        
        self.to(self.device)

    def run(self, x_batch, cur_batch_size=None, record=False):
        print("=" * 50)
        print("RUN")
        if cur_batch_size is None:
            cur_batch_size = len(x_batch)
        self.reset_states(cur_batch_size)
        self.input_group.feed_data(x_batch) #[250, 500, 96]
        lower_bounds = []
        for t in range(0, self.nb_time_steps, self.n_keys): # the interpolation change here
            print(f"Time step: {t}")
            stork.nodes.base.CellGroup.clk = t
            if self.n_keys == 1:
                self.evolve_all()
                self.propagate_all()
            else:
                if t == 0:
                    self.evolve_all()
                    self.propagate_all()
                    for g in self.groups[1:]:
                        # print("Lower Bound")
                        # For my implementation just two group, lif and readout
                        # (250, 64), (250, 2)
                        # print(f"Group Input: {g.input.shape}") 
                        lower_bounds.append(g.input.clone()) # got all lower bound
                        # print(f"Lower Bounds: {len(lower_bounds)}")
                else:
                    # Evolve the input groups
                    input_group = self.groups[0]
                    input_group.evolve()
                    input_group.clear_input()

                    # TODO: Miss connection so that the upper threshold == lower threshold (dont forget to exclude the upper threshold so that it used on the next key step)
                    for i, g in enumerate(self.groups[1:]):
                        lower_bound = lower_bounds[i]
                        upper_bound = g.input.clone()
                        # print(f"Lower Bound Shape: {lower_bound.shape}") # (250, 64)
                        # print(f"Upper Bound Shape: {upper_bound.shape}") # (250, 64)
                        interpolation = self.interpolate(lower_bound, upper_bound, self.n_keys)
                        print(f"Lower Bound: {lower_bound}")
                        print(f"Upper Bound: {upper_bound}")
                        print(f"Interpolation: {interpolation}")
                        print(f"Interpolation Shape: {interpolation.shape}")
                        for input in interpolation:
                            print('Checking for the changing input')
                            print(f"Input: {g.input}")
                            print(f"Input Shape: {g.input.shape}")
                            g.add_to_state(g.default_target, input)
                            print(f"Input: {g.input}")
                            print(f"Input Shape: {g.input.shape}")
                            g.evolve()
                            g.clear_input()
                            print(f"Output: {g.out}")
                            print(f"Output Shape: {g.out.shape}")

                            raise ValueError("This is breakpoint")
                        lower_bounds[i] = upper_bound
                    
            self.execute_all()
            if record:
                self.monitor_all()
        self.out = self.output_group.get_out_sequence()
        print("RUN=" * 50)
        return self.out

    def interpolate(self, A, B, n_steps):
        device = A.device
        alphas = torch.linspace(0, 1, n_steps+1, device=device).view(-1, 1, 1)

        interpolated = torch.lerp(A, B, alphas)

        return interpolated # it should be (n_steps + 1, 250, 64)


        