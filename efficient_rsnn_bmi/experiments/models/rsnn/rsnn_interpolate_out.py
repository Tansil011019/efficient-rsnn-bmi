import torch
import torch.nn.functional as F

import stork
from stork.models import (
    loss_stacks,
    generators
)
import numpy as np
from tqdm import tqdm
import time
import json

from .rsnn import BaselineRecurrentSpikingModel

class InterpolateOutRecurrentSpikingModel(BaselineRecurrentSpikingModel):
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

    # def run(self, x_batch, cur_batch_size=None, record=False):
    #     if cur_batch_size is None:
    #         cur_batch_size = len(x_batch)
    #     self.reset_states(cur_batch_size)
    #     self.input_group.feed_data(x_batch)

    #     time_steps_to_run = list(range(0, self.nb_time_steps, self.n_keys))
    #     final_steps = self.nb_time_steps - 1
    #     if final_steps >= 0 and (not time_steps_to_run or time_steps_to_run[-1] != final_steps):
    #         time_steps_to_run.append(final_steps)

    #     # print(time_steps_to_run)
    #     # print(len(time_steps_to_run))

    #     for t in time_steps_to_run:
    #         stork.nodes.base.CellGroup.clk = t
    #         self.evolve_all() # Assume this part is true
    #         self.propagate_all()
    #         self.execute_all()
    #         if record:
    #             self.monitor_all()
    #     output = self.output_group.get_out_sequence()
    #     self.out = self.interpolate_out(output)
    #     return self.out

    def run(self, x_batch, cur_batch_size=None, record=False):
        if cur_batch_size is None:
            cur_batch_size = len(x_batch)
        self.reset_states(cur_batch_size)
        self.input_group.feed_data(x_batch)
        for t in range(0, self.nb_time_steps, self.n_keys):
            stork.nodes.base.CellGroup.clk = t
            self.evolve_all() # Assume this part is true
            self.propagate_all()
            self.execute_all()
            if t + self.n_keys > self.nb_time_steps and t < self.nb_time_steps-1:
                print(t)
                stork.nodes.base.CellGroup.clk = self.nb_time_steps - 1
                self.evolve_all() # Assume this part is true
                self.propagate_all()
                self.execute_all()
            if record:
                self.monitor_all()
        output = self.output_group.get_out_sequence()
        # print("Enter interpolation")
        self.out = self.interpolate_out(output)
        # print("Exit interpolation")
        return self.out
    
    def interpolate(self, A, B, n_steps):
        # print("=" * 50)
        # print(f"Interpolating between {A.shape} and {B.shape} with {n_steps} steps")
        # print(A)
        # print(B)
        device = A.device
        alphas = torch.linspace(0, 1, n_steps+1, device=device).view(-1, 1)
        # print(f"Alphas shape: {alphas.shape}, Alphas: {alphas}")

        interpolated = torch.lerp(A, B, alphas.to(self.dtype))
        # print("=" * 50)

        return interpolated # it should be (n_steps + 1, 250, 64)

    # def interpolate_out(self, output):
    #     """
    #     Interpolate the output using linear interpolation.
    #     The output is expected to be of shape (batch_size, nb_time_steps, nb_outputs).
    #     """
    #     if self.n_keys == 1:
    #         return output
        
    #     batch_size, nb_time_steps, nb_outputs = output.shape # [250, 64, 2]
    #     interpolated_output = torch.zeros((batch_size, self.nb_time_steps, nb_outputs), device=output.device, dtype=output.dtype)
    #     # print(interpolated_output.shape)
    #     print(output.shape)

    #     for i in range(batch_size):
    #         for j in range(nb_time_steps):
    #             if j == nb_time_steps - 1:
    #                 # If we are at the last time step, just copy the output
    #                 interpolated_output[i, -1] = output[i, j]
    #             else:
    #                 start_idx = j * self.n_keys
    #                 end_idx = start_idx + self.n_keys

    #                 # handle edge cases
    #                 if end_idx > self.nb_time_steps:
    #                     end_idx = self.nb_time_steps

    #                 actual_steps = end_idx - start_idx

    #                 lower_bound = output[i, j]
    #                 upper_bound = output[i, j + 1] 
    #                 interpolated_output[i, start_idx:end_idx] = self.interpolate(lower_bound, upper_bound, actual_steps)[:-1]
    #             # print(f"Interpolating for batch {i}, time step {j}: start_idx={start_idx}, end_idx={end_idx}, lower_bound={lower_bound}, upper_bound={upper_bound}")
    #             # interpolated_output[i, start_idx:end_idx] = output[i, j].unsqueeze(0).repeat(self.n_keys, 1)
    #             # print(output[i, j])
    #             # print(i, j)
    #             # print(start_idx, end_idx, interpolated_output[i, start_idx:end_idx].shape)
    #             # print(interpolated_output[i, start_idx:end_idx])
    #             # if j == 3:
    #                 # raise ValueError("Breakpoint")
    #     return interpolated_output

    def interpolate_out(self, outputs):
        if self.n_keys == 1: return outputs

        # print(outputs)
        transposed_outputs = outputs.transpose(1, 2)

        interpolated_transposed = F.interpolate(
            transposed_outputs,
            size=self.nb_time_steps,
            mode='linear',
            align_corners=True
        )

        interpolated_transposed = interpolated_transposed.transpose(1, 2)

        # print(interpolated_transposed)

        return interpolated_transposed
    
    def evaluate(self, test_dataset, train_mode=False):
            self.train(train_mode)
            self.prepare_data(test_dataset)
            metrics = []
            data_iter = self.data_generator(test_dataset, shuffle=False)
            for local_X, local_y in tqdm(data_iter, desc="Evaluating", total=len(data_iter)):
                output = self.forward_pass(local_X, cur_batch_size=len(local_X))
                # print(f"Output shape: {output.shape}, Local y shape: {local_y.shape}")
                # interpolated_output = self.interpolate_out(output)
                # print(interpolated_output.shape, local_y.shape)
                total_loss = self.get_total_loss(output, local_y)
                # store loss and other metrics
                metrics.append(
                    [self.out_loss.item(), self.reg_loss.item()] + self.loss_stack.metrics
                )

            return np.mean(np.array(metrics), axis=0)
        
    def train_epoch(self, dataset, shuffle=True):
        self.train(True)
        self.prepare_data(dataset)
        metrics = []
        data_iter = self.data_generator(dataset, shuffle=shuffle)
        for local_X, local_y in tqdm(data_iter, desc="Training", total=len(data_iter)):
            output = self.forward_pass(local_X, cur_batch_size=len(local_X))
            total_loss = self.get_total_loss(output, local_y)

            # store loss and other metrics
            metrics.append(
                [self.out_loss.item(), self.reg_loss.item()] + self.loss_stack.metrics
            )

            # Use autograd to compute the backward pass.
            self.optimizer_instance.zero_grad()
            total_loss.backward()

            self.optimizer_instance.step()
            self.apply_constraints()
            
        if self.scheduler_instance is not None:
            self.scheduler_instance.step()

        return np.mean(np.array(metrics), axis=0)
    
    def fit_validate(
        self, dataset, valid_dataset, best_model_path=None, nb_epochs=10, verbose=True, wandb=None, early_stop=False, patience=10
    ):
        self.hist_train = []
        self.hist_valid = []
        self.wall_clock_time = []

        assert early_stop and best_model_path is not None, "The best model path should not empty"

        if early_stop:
            epochs_no_improve = 0
            best_val_loss = np.inf
            best_model_path = best_model_path

            print(f"Early stopping enabled with patience of {patience}. Best model will be saved to '{best_model_path}'")

        for ep in range(nb_epochs):
            t_start = time.time()
            self.train()
            ret_train = self.train_epoch(dataset)

            self.train(False)
            ret_valid = self.evaluate(valid_dataset)
            self.hist_train.append(ret_train)
            self.hist_valid.append(ret_valid)

            # Early Stopping
            current_val_loss = ret_valid[0]
            if current_val_loss < best_val_loss:
                best_val_loss  = current_val_loss
                epochs_no_improve = 0
                torch.save(self.state_dict(), best_model_path)
                if verbose: print(f"Validation loss improved to {best_val_loss:.6f}. Saving model.")
            else:
                epochs_no_improve += 1
                if verbose: print(f"Validation loss did not improve. Patience: {epochs_no_improve}/{patience}")

            if self.wandb is not None:
                self.wandb.log(
                    {
                        key: value
                        for (key, value) in zip(
                            self.get_metric_names()
                            + self.get_metric_names(prefix="val_"),
                            ret_train.tolist() + ret_valid.tolist(),
                        )
                    }
                )

            if verbose:
                t_iter = time.time() - t_start
                self.wall_clock_time.append(t_iter)
                print(
                    "%02i %s --%s t_iter=%.2f"
                    % (
                        ep,
                        self.get_metrics_string(ret_train),
                        self.get_metrics_string(ret_valid, prefix="val_"),
                        t_iter,
                    )
                )

            if epochs_no_improve >= patience:
                print(f"\nEarly stopping triggered after {patience} epochs with no improvement.")
                break

        self.hist = np.concatenate(
            (np.array(self.hist_train), np.array(self.hist_valid))
        )
        self.fit_runs.append(self.hist)
        dict1 = self.get_metrics_history_dict(np.array(self.hist_train), prefix="")
        dict2 = self.get_metrics_history_dict(np.array(self.hist_valid), prefix="val_")
        history = {**dict1, **dict2}
        return history