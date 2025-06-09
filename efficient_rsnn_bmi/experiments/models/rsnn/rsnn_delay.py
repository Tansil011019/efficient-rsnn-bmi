import torch   
import numpy as np
from tqdm import tqdm
import time

from DCLS.construct.modules import Dcls1d
import stork
from stork.models import (
    RecurrentSpikingModel,
    loss_stacks,
    generators
)
from efficient_rsnn_bmi.base.delays.delay import CustomDelayConnection

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
        self.dcls_connection = [
            conn
            for conn in self.connections
            if isinstance(conn.op, Dcls1d)
        ]

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
            if isinstance(o, CustomDelayConnection):
                if self.conv_out_size is None:
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

    def decrease_sig(self, cur_ep, final_ep):
        '''
        Decrease the sig value along with the increase of the epoch
        '''
        alpha = 0

        if not self.dcls_connection:
            raise ValueError("No Dcls1d connection found in the model.")

        last_dcls_con = self.dcls_connection[-1]

        sig_stop_ep = last_dcls_con.sig_stop_ep
        sig_threshold = last_dcls_con.sig_threshold
        sig_init = last_dcls_con.sig_init
        sig = last_dcls_con.op.SIG[0, 0, 0, 0].detach().cpu().item()

        dec_end_ep = final_ep * sig_stop_ep

        if dec_end_ep <= 0:
            raise ValueError("dec_end_ep must be greater than 0.")

        if cur_ep < int(dec_end_ep) and sig > sig_threshold:
            alpha = (sig_threshold / sig_init) ** (1 / dec_end_ep)  
            for conn in self.dcls_connection:
                if isinstance(conn.op, Dcls1d): conn.decrease_sig(alpha)
    
    def configure_optimizer(self, optimizer_class, optimizer_kwargs):
        '''
        Configures the optimizer with support for different learning rates for 'P' parameters.
        '''      
        if optimizer_kwargs is not None:
            kwargs = optimizer_kwargs.copy()

            base_lr = kwargs.pop("lr", 1e-3) # 1e-3 is the one implemented in the delay paper
            lr_P = kwargs.pop("lr_P", base_lr)

            delay_pos_param = []
            other_param = []

            for name, param in self.named_parameters():
                if "SIG" in name and param.requires_grad:
                    param.requires_grad = False
                if param.requires_grad:
                    if 'P' in name:
                        delay_pos_param.append(param)
                    else:
                        other_param.append(param)
            
            param_groups = [
                {
                    "params": other_param, "lr": base_lr,
                },
                {
                    "params": delay_pos_param, "lr": lr_P,
                }
            ]
            self.optimizer_instance = optimizer_class(param_groups, **kwargs)
        else:
            self.optimizer_instance = optimizer_class(self.parameters())

    def train_epoch(self, dataset, shuffle=True):
        self.train(True)
        self.prepare_data(dataset)
        metrics = []
        data_iter = self.data_generator(dataset, shuffle=shuffle)
        for local_X, local_y in tqdm(data_iter, desc="Training", total=len(data_iter)):
            output = self.forward_pass(local_X, cur_batch_size=len(local_X))
            if output.shape[1] > local_y.shape[1]:
                output = output[:, :local_y.shape[1], :]
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
        self.pos_logs = []
        self.wall_clock_time = []

        assert early_stop and best_model_path is not None, "The best model path should not empty"
        
        pos_val = [
           np.copy(conn.op.P.detach().cpu().numpy()) 
           for conn in self.dcls_connection
        ]
        pre_pos_epoch = pos_val.copy()
        pre_pos_5epochs = pos_val.copy()

        if early_stop:
            epochs_no_improve = 0
            best_val_loss = np.inf
            best_model_path = best_model_path

            print(f"Early stopping enabled with patience of {patience}. Best model will be saved to '{best_model_path}'")

        for ep in range(nb_epochs):
            t_start = time.time()
            self.train()
            ret_train = self.train_epoch(dataset) # 1 epoch = 2 min
            self.decrease_sig(ep, nb_epochs)
            # Just monitoring the changes from the delay position (same as the references)
            pos_logs = {}
            for i, conn in enumerate(self.dcls_connection):
                curr_pos = conn.op.P.detach().cpu().numpy()
                dpos_epoch = np.abs(curr_pos - pre_pos_epoch[i]).mean()
                pos_logs[f'dpos{i}_epoch'] = dpos_epoch
                pre_pos_epoch[i] = curr_pos.copy()

                if ep % 5 == 0 and ep > 0:
                    dpos_5epochs = np.abs(curr_pos - pre_pos_5epochs[i]).mean()
                    pos_logs[f'dpos{i}_5epochs'] = dpos_5epochs
                    pre_pos_5epochs[i] = curr_pos.copy()

            self.train(False)
            ret_valid = self.evaluate(valid_dataset)
            self.hist_train.append(ret_train)
            self.hist_valid.append(ret_valid)
            self.pos_logs.append(pos_logs)

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
        history = {
            **dict1, **dict2, "pos_logs": self.pos_logs
        }
        return history