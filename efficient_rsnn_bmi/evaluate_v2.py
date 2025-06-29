from omegaconf import DictConfig, OmegaConf
import hydra
from pathlib import Path

import numpy as np
import torch
from datetime import datetime

from efficient_rsnn_bmi.utils.logger import get_logger

from efficient_rsnn_bmi.core.dataloader import get_dataloader, compute_input_firing_rates
from efficient_rsnn_bmi.core.model import get_model
from efficient_rsnn_bmi.core.train import configure_model, train_validate_model  
from efficient_rsnn_bmi.neurobench.wrappers.stork_wrapper import StorkModel
from neurobench.metrics.static import (
    Footprint,
    ConnectionSparsity,
)
from neurobench.benchmarks import Benchmark
from efficient_rsnn_bmi.neurobench.metrics.activation_sparsity import ActivationSparsity
from efficient_rsnn_bmi.neurobench.metrics.r2 import R2
from efficient_rsnn_bmi.neurobench.metrics.syn_ops import SynapticOperations

def evaluate_model(model, test_dataset):
    with torch.no_grad():
        model.train(False)
        model.prepare_data(test_dataset)

        total_squared_error = 0.0
        total_points = 0.0
        all_preds = []
        all_target = []
        for local_X, local_y in model.data_generator(test_dataset, shuffle=False):
            output = model.forward_pass(local_X, cur_batch_size=len(local_X))
            if output.shape[1] > local_y.shape[1]:
                output = output[:, :local_y.shape[1], :]
            
            # print(f"Output shape: {output.shape}")
            # print(f"Output type: {output.dtype}")
            # print(f"Output device: {output.device}")
            # print(f"Label shape: {local_y.shape}")
            # print(f"Label type: {local_y.dtype}")
            # print(f"Label device: {local_y.device}")

            local_y = local_y.to(output.device)
            
            squared_error = (output - local_y) ** 2
            # print(f"Squared error: {squared_error}")
            total_squared_error += torch.sum(squared_error).detach().cpu().item()
            total_points += output.numel()

            all_preds.append(output.reshape(-1, 2).detach().cpu())
            all_target.append(local_y.reshape(-1, 2).detach().cpu())

        preds = torch.cat(all_preds, dim=0)
        targets = torch.cat(all_target, dim=0)

        x_corr = torch.corrcoef(torch.stack([preds[:, 0], targets[:, 0]]))[0, 1].item()
        y_corr = torch.corrcoef(torch.stack([preds[:, 1], targets[:, 1]]))[0, 1].item()

        rmse = (total_squared_error / total_points) ** 0.5

    return rmse, x_corr, y_corr


@hydra.main(version_base=None, config_path="../config", config_name="config")
def evaluate(config: DictConfig) -> None:
    print(OmegaConf.to_yaml(config))

    experiment_name = config.experiments.name
    output_dir = Path("pretrain_results") / experiment_name
    seed = config.seed
    logger = get_logger(experiment_name.capitalize())
    logger.info(f"Starting {experiment_name.capitalize()} Evaluation...")

    device = config.experiments.device
    dtype = getattr(torch, config.dtype)
    dataloader = get_dataloader(config.datasets, dtype=dtype)

    monkey_name = config.train_monkey # Train it one by one, my gpu cannot bare with all
    nb_inputs = config.datasets.nb_inputs[monkey_name]
    nb_time_steps = int(config.datasets.sample_duration / config.datasets.dt)
    nb_outputs = config.datasets.nb_outputs
    dt = config.datasets.dt

    max_delay = None
    if config.experiments.name == "synaps-delay":
        max_delay = config.experiments.max_delays // dt
        max_delay = max_delay if max_delay % 2 == 1 else max_delay + 1

    logger.info("Collecting Data")
    filenames = config.eval_data
    train_data, _, test_data = dataloader.get_single_session_data(filenames)

    logger.info("Initializing Model")
    if train_data is not None:
        mean1, mean2 = compute_input_firing_rates(
            train_data, config.datasets
        )
    else:
        mean1 = None

    model = get_model(
        config.experiments, 
        nb_inputs=nb_inputs, 
        nb_outputs=nb_outputs,
        nb_time_steps=nb_time_steps,
        dt=dt,
        dtype=dtype,
        device=device,
        input_firing_rates=(mean1, mean2),
        max_delay = max_delay,
        verbose=True
    )

    logger.info("Configuring Model")
    model = configure_model(
        model,
        config.experiments,
        nb_time_steps=nb_time_steps,
        dt=dt,
        dtype=dtype,
        seed=seed
    )

    loaded_model_state = torch.load(config.eval_model)
    model.load_state_dict(loaded_model_state)

    model = model.half()
    test_data.dtype = torch.float16

    #Configure to test data
    model.set_nb_steps(test_data[0][0].shape[0]) # I hardcode it for easier implementation

    # Evaluate model
    # rmse, x_corr, y_corr = evaluate_model(model, test_data)
    # avg_corr = (x_corr + y_corr) / 2
    # print(f"RMSE: {rmse:.4f}")
    # print(f"Pearson Correlation - X: {x_corr:.4f}, Y: {y_corr:.4f},  Mean: {avg_corr:.4f}")

    #Benchmark model
    stork_model = StorkModel(model)

    test_set_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=1,
        shuffle=False
    )

    # static_metrics = [Footprint, ConnectionSparsity]
    static_metrics = []
    workload_metrics = [SynapticOperations]
    # workload_metrics = [R2, ActivationSparsity, SynapticOperations]

    benchmark = Benchmark(
        stork_model,
        test_set_loader,
        [],
        [],
        [static_metrics, workload_metrics],
    )

    with torch.no_grad():
        result = benchmark.run(verbose=True, device=device)
    logger.info(f"Benchmark Result: {result}")

    benchmark.save_benchmark_results("result", "json") # I still change this manually

if __name__ == "__main__":
    evaluate()