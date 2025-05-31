from omegaconf import DictConfig, OmegaConf
import hydra
from pathlib import Path

import numpy as np
import torch
from datetime import datetime

import os
import json

from efficient_rsnn_bmi.utils.logger import get_logger
from efficient_rsnn_bmi.utils.misc import convert_np_float_to_float
from efficient_rsnn_bmi.utils.state import save_model_state, load_model_state

from efficient_rsnn_bmi.core.dataloader import get_dataloader, compute_input_firing_rates
from efficient_rsnn_bmi.core.model import get_model
from efficient_rsnn_bmi.core.train import configure_model, train_validate_model  

date = datetime.now().strftime("%Y-%m-%d")
time = datetime.now().strftime("%H-%M-%S")

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(config: DictConfig) -> None:
    print(OmegaConf.to_yaml(config))

    experiment_name = config.experiments.name
    output_dir = Path("states") / experiment_name
    logger = get_logger(experiment_name.capitalize())
    logger.info(f"Starting {experiment_name.capitalize()} Experiments...")

    if config.seed:
        seed = config.seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        if torch.cuda.is_available and 'cuda' in config.experiments.device:
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        os.environ['PYTHONHASHSEED'] = str(seed)

    device = config.experiments.device
    dtype = getattr(torch, config.dtype)
    dataloader = get_dataloader(config.datasets, dtype=dtype)

    for monkey_name in config.train_monkeys:
        nb_inputs = config.datasets.nb_inputs[monkey_name]
        nb_time_steps = int(config.datasets.sample_duration / config.datasets.dt)
        nb_outputs = config.datasets.nb_outputs
        dt = config.datasets.dt
        
        # I just add the conditional for delay here, which not really good, open for improvement
        max_delay = None
        if config.experiments.name == "synaps-delay":
            max_delay = config.experiments.max_delays // dt
            max_delay = max_delay if max_delay % 2 == 1 else max_delay + 1

        # * Phase 1: Pretraining
        if config.pretraining:
            logger.info("Phase 1: Pretraining")
            logger.info("=" * 50)
            
            # 1.1 Data
            logger.info("Phase 1.1: Collecting Data")
            filenames = list(config.datasets.pretrain_filenames[monkey_name].values())
            pretrain_data, pretrain_val_data, pretrain_test_data = dataloader.get_multiple_sessions_data(filenames)

            # 1.2 Initialize Model
            logger.info("Phase 1.2: Initializing Model")
            if pretrain_data is not None:
                mean1, mean2 = compute_input_firing_rates(
                    pretrain_data, config.datasets
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

            # 1.3 Configure Model
            logger.info("Phase 1.3: Configuring Model")
            model = configure_model(
                model,
                config.experiments,
                nb_time_steps=nb_time_steps,
                dt=dt,
                dtype=dtype,
                seed=seed
            )

            # 1.4 Pretraining
            logger.info(f"Phase 1.4: Pretraining On All {monkey_name.capitalize()} Sessions...")
            model, history = train_validate_model(
                model,
                config.experiments,
                pretrain_data,
                pretrain_val_data,
                nb_epochs = config.experiments.training.nb_epochs_pretrain,
                verbose=True,
                snapshot_prefix=output_dir/f"pretrained/{date}/{time}/pretrained_on_{monkey_name}_"
            )

            results = {}
            for k, v in history.items():
                if "val" in k :
                    results[k] = v.tolist()
                elif k == 'pos_logs':
                    results[k] = v
                else: 
                    results["train_" + k] = v.tolist()

            logger.info("Phase 1 Completed")

            # Saving pretraining history in json file
            logger.info("Saving Pretraining History")
            converted_result = convert_np_float_to_float(results)
            with open(output_dir/f"pretrained/{date}/{time}/pretraining_results_on_{monkey_name}.json", "w") as f:
                json.dump(converted_result, f, indent=4)

            # Saving pretrain model state
            logger.info("Saving Pretrained Model State")
            save_model_state(model, output_dir/f"pretrained/{date}/{time}/pretraining_on_{monkey_name}.pth")

            pretrained_model = model.state_dict()

        elif config.load_state[monkey_name]:
            logger.info(f"Phase 1: Loading Pretrained Model For {monkey_name.capitalize()}")
            logger.info("=" * 50)
            pretrained_model = load_model_state(config.load_state[monkey_name])
            logger.info("Model State Loaded.")

        else:
            logger.info("No Pretraining or Model State Loaded.")
            pretrained_model = None
            



if __name__ == "__main__":
    main()