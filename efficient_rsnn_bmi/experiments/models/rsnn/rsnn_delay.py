import torch   
import numpy as np
from tqdm import tqdm

import stork
from stork.models import (
    RecurrentSpikingModel,
    loss_stacks,
    generators
)
from efficient_rsnn_bmi.base.delay import CustomDelayConnection

class DelayRecurrentSpikingModel(RecurrentSpikingModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_out_size= None
    
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
            if isinstance(o, CustomDelayConnection) and self.conv_out_size is None:
                self.conv_out_size = o.conv_out_size
                if not self.conv_out_size.is_integer() or self.conv_out_size <= 0:
                    raise ValueError(f"Convolution Output Size must be positive integer, got {self.conv_out_size}")
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
        if cur_batch_size is None:
            cur_batch_size = len(x_batch)
        self.reset_states(cur_batch_size) # reset group -> decay
        self.delay_add_step = int(self.conv_out_size - 1)
        self.input_group.feed_data(x_batch, self.delay_add_step) # local data -> (250, 500, 96) just change the in channel to the shape
        nb_iter_ts = int(self.nb_time_steps + self.delay_add_step)
        for t in range(nb_iter_ts):
            stork.nodes.base.CellGroup.clk = t
            self.evolve_all()
            self.propagate_all()
            self.execute_all()
            if record:
                self.monitor_all()
        self.out = self.output_group.get_out_sequence()
        return self.out

    def reset_states(self, batch_size=None):
        for g in self.groups:
            g.reset_state(batch_size)
        for c in self.connections:
            if hasattr(c, "reset_states"):
                c.reset_states()

    def evaluate(self, test_dataset, train_mode=False):
        self.train(train_mode)
        self.prepare_data(test_dataset)
        metrics = []
        data_iter = self.data_generator(test_dataset, shuffle=False)
        for local_X, local_y in tqdm(data_iter, desc="Evaluating", total=len(data_iter)):
            # print(f"Local X: {local_X}")
            # print(f"Local X Shape: {local_X.shape}") # (250, 500, 96) -> (batch size, nb time steps, in chann)
            output = self.forward_pass(local_X, cur_batch_size=len(local_X))
            # Trimming the result 
            # It’s like predicting the next words in a sentence — if the real sentence ends at word 10, we don’t
            # want to grade the model on words 11–13 it generated just to finish processing its internal memory.
            if output.shape[1] > local_y.shape[1]:
                output = output[:, :local_y.shape[1], :]
            total_loss = self.get_total_loss(output, local_y)   
            # store loss and other metrics
            metrics.append(
                [self.out_loss.item(), self.reg_loss.item()] + self.loss_stack.metrics
            )

        return np.mean(np.array(metrics), axis=0)