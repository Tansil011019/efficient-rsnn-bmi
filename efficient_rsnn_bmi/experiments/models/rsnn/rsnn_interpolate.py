import torch
import stork
from stork.models import (
    loss_stacks,
    generators
)
import numpy as np
from tqdm import tqdm
import time

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
        if cur_batch_size is None:
            cur_batch_size = len(x_batch)
        self.reset_states(cur_batch_size)
        self.input_group.feed_data(x_batch) #[250, 500, 96]
        lower_bounds = []
        for t in range(0, self.nb_time_steps, self.n_keys): # the interpolation change here
            stork.nodes.base.CellGroup.clk = t
            self.forward_interpolation(t, lower_bounds=lower_bounds, record=record)

        # for the left over
        max_t = self.nb_time_steps - 1
        remaining = (self.nb_time_steps - 1) % self.n_keys
        if (remaining != 0):
            self.forward_interpolation(max_t, remaining, lower_bounds, record)
            
        self.out = self.output_group.get_out_sequence()
        return self.out

    def interpolate(self, A, B, n_steps):
        device = A.device
        alphas = torch.linspace(0, 1, n_steps+1, device=device).view(-1, 1, 1)

        interpolated = torch.lerp(A, B, alphas)

        return interpolated # it should be (n_steps + 1, 250, 64)
    
    def forward_interpolation(self, t, key=None, lower_bounds=[], record=False):
        keys = key if key else self.n_keys
        if keys == 1:
                self.evolve_all()
                self.propagate_all()
        else:
            if t == 0:
                self.evolve_all()
                self.propagate_all()
                for g in self.groups[1:]:
                    lower_bounds.append(g.input.clone()) # got all lower bound
            else:
                self.evolve_all() # already evolve the 0.4438 one

                # Update first input
                first_con = self.connections[0]
                first_con.propagate()

                for i, g in enumerate(self.groups[1:]):
                    lower_bound = lower_bounds[i]
                    upper_bound = g.input.clone()

                    interpolation = self.interpolate(lower_bound, upper_bound, keys)
                    g.states['input'] = interpolation[1].clone()
                    g.input = g.states['input'] # change the input to current interpolation time step

                    for input in interpolation[2:]: # update till before upper bound
                        g.evolve()
                        g.clear_input()
                        g.states['input'] = input.clone()
                        g.input = g.states['input']

                    connection = self.connections[i+1]
                    connection.propagate() # update the next input 
                    lower_bounds[i] = upper_bound

        self.execute_all()
        if record:
            self.monitor_all()

    def evaluate(self, test_dataset, train_mode=False):
            self.train(train_mode)
            self.prepare_data(test_dataset)
            metrics = []
            data_iter = self.data_generator(test_dataset, shuffle=False)
            for local_X, local_y in tqdm(data_iter, desc="Evaluating", total=len(data_iter)):
                output = self.forward_pass(local_X, cur_batch_size=len(local_X))
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