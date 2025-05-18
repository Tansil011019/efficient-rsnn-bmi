from omegaconf import DictConfig, OmegaConf
import hydra
from pathlib import Path
from hydra.utils import to_absolute_path

import json
import torch
import numpy as np
from datetime import datetime

from efficient_rsnn_bmi.core.dataloader import get_dataloader
from efficient_rsnn_bmi.core.model import get_model
from efficient_rsnn_bmi.core.train import configure_model, train_validate_model, prune_retrain_model_iterate
from efficient_rsnn_bmi.core.evaluate import evaluate_model

from efficient_rsnn_bmi.utils.logger import get_logger
from data.config.dataloader import DatasetLoaderConfig

from efficient_rsnn_bmi.utils.helper import from_config
from efficient_rsnn_bmi.utils.misc import convert_np_float_to_float
from efficient_rsnn_bmi.utils.state import save_model_state, load_model_state
from efficient_rsnn_bmi.utils.plotting import plot_training, plot_cumulative_mse

logger = get_logger("train-baselineRSNN")

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_dir = Path("outputs") / "baseline" / timestamp

@hydra.main(version_base=None, config_path="../config", config_name="defaults")
def train_rsnn_tiny(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    logger.info("Starting new simulation...")

    dtype = getattr(torch, cfg.dtype)

    # to make the experiment consistent, the initial weight will be make fixed
    if cfg.seed:
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        # if using GPU, make sure initialize the weight that is used by GPU
        if torch.cuda.is_available and cfg.node:
            torch.cuda.manual_seed(cfg.seed)
            # Ensure exact repeatability
            torch.backends.cudnn.deterministic = True # Ensures always using the same computation method
            torch.backends.cudnn.benchmark = False # Disables cuDNN’s automatic selection of the fastest computation method

    logger.info(f"Config: {cfg}")
    # Get DataLoader
    dataloader = get_dataloader(cfg, dtype=dtype)

    for monkey_name in cfg.train_monkeys:
        nb_inputs = cfg.datasets.nb_inputs[monkey_name]
        logger.info(f"Training on monkey: {monkey_name}")

        if cfg.pretraining:
            logger.info("=" * 50)
            logger.info("Pretraining on " + monkey_name + "...")
            logger.info("=" * 50)

            filename = list(cfg.datasets.pretrain_filenames[monkey_name].values())
            logger.info(f"Loading pretraining data for {filename} ...")
            
            logger.info("Constructing model for " + monkey_name + " pretraining...")
            pretrain_dat, pretrain_val_dat, _ = dataloader.get_multiple_sessions_data(
                filename
            )

            model = get_model(cfg, nb_inputs=nb_inputs, dtype=dtype, data=pretrain_dat)
            logger.info(f"Model constructed. {model}")

            logger.info("Configuring model...")
            model = configure_model(model, cfg, pretrain_dat, dtype)
            logger.info(f"Model configured. {model}")

            logger.info("Pretraining on all {} sessions...".format(monkey_name))
            pretraining_snapshot_prefix = output_dir / f"pretraining/baselineRSNN_pretrain_{monkey_name}_"
            model, history = train_validate_model(
                model,
                cfg,
                pretrain_dat,
                pretrain_val_dat,
                cfg.training.nb_epochs_pretrain,
                verbose=cfg.training.verbose,
                snapshot_prefix=pretraining_snapshot_prefix,
            )
            results = {}
            for k, v in history.items():
                if "val" in k:
                    results[k] = v.tolist()
                else:
                    results["train_" + k] = v.tolist()

            logger.info("Pretraining complete.")

            converted_results = convert_np_float_to_float(results)
            with open(f"{output_dir}/baselineRSNN-results-pretraining-{monkey_name}.json", "w") as f:
                json.dump(converted_results, f, indent=4)
            
            save_model_state(model, f"{output_dir}/baselineRSNN-results-pretraining-{monkey_name}.pth")

            pretrained_model = model.state_dict()

        elif cfg.load_state[monkey_name]:
            logger.info("Loading pretrained model for " + monkey_name)
            pretrained_model = load_model_state(cfg.load_state[monkey_name])
            logger.info("Model state loaded.")

        else:
            logger.info("No pretraining or model state loaded.")
            pretrained_model = None


        for session_name, filename in cfg.datasets.filenames[monkey_name].items():
            logger.info("=" * 50)
            logger.info("Training on " + session_name + "...")
            logger.info("=" * 50)

            logger.info("Constructing model for " + session_name + "...")

            train_dat, val_dat, test_dat = dataloader.get_single_session_data(filename)
            # print(f"train_dat: {train_dat[0]}, \nval_dat: {val_dat[0]}, \ntest_dat: {test_dat[0]}")
            model = get_model(cfg, nb_inputs=nb_inputs, dtype=dtype, data=train_dat)

            logger.info("Configuring model...")
            model = configure_model(model, cfg, train_dat, dtype)
            
            # Load pretrained model state
            if pretrained_model is not None:
                model.load_state_dict(pretrained_model)
                logger.info("Pretrained model state loaded.")

            logger.info("Training on " + session_name + "...")
            training_snapshot_prefix = output_dir / f"training/baselineRSNN_{session_name}_"
            model, history = train_validate_model(
                model,
                cfg,
                train_dat,
                val_dat,
                cfg.training.nb_epochs_train,
                verbose=cfg.training.verbose,
                snapshot_prefix=training_snapshot_prefix,
            )

            results = {}
            for k, v in history.items():
                if "val" in k:
                    results[k] = v.tolist()
                else:
                    results["train_" + k] = v.tolist()

            logger.info("Training complete.")

        save_model_state(model,f"{output_dir}/baselineRSNN-results-training-{session_name}.pth")

        if cfg.training.is_prune:
            model = prune_retrain_model_iterate(
                model,
                cfg,
                train_dat,  
                val_dat,
                logger,
                history['r2'][-1],
                history['val_r2'][-1],
                nb_epochs_retrain=cfg.training.nb_epochs_retrain,
                prune_percentage_start=cfg.training.prune_percentage_start,
                tolerance=cfg.training.tolerance, 
                prune_precision=cfg.training.prune_precision,
                max_prune_percentage=cfg.training.max_prune_percentage,
                is_plot_pruning=cfg.training.is_plot_pruning,
                is_pruning_ver=cfg.training.is_pruning_ver,
                session_name=session_name,
                pruning_plot_prefix=output_dir / f"pruning/baselineRSNN_pruning_{session_name}_",
            )
            save_model_state(model, f"{output_dir}/baselineRSNN-results-pruning-{session_name}.pth")
        
        if cfg.model.is_half:
            model = model.half()
            # Save pruned model state
            if cfg.training.is_prune:
                save_model_state(model, f"{output_dir}/baselineRSNN-results-half-pruning-{session_name}.pth")
            else:
                save_model_state(model, f"{output_dir}/baselineRSNN-results-half-{session_name}.pth")
                
            logger.info("Model converted to half precision.")

            test_dat.dtype = torch.float16
            logger.info("Test data converted to half precision.")
        
        if cfg.seed:
            path = Path(to_absolute_path('models')) / session_name
            path.mkdir(parents=True, exist_ok=True)
            filepath = path / ("baselineRSNN-" + str(cfg.seed) + ".pth")
            save_model_state(model, filepath)
            
        logger.info("Saved model state.")

        logger.info("=" * 50)
        logger.info("Evaluating model...")
        logger.info("=" * 50)

        if cfg.plotting.plot_cumulative_mse:
            fig, ax = plot_cumulative_mse(
                model, val_dat, save_path=output_dir / "plotting/baselineRSNN_cumulative_se_" + session_name + ".png"
            )

        model, scores, pred, bm_results = evaluate_model(model, cfg, test_dat)

        logger.info("Benchmark results:")
        for k, v in bm_results.items():
            # log key and value rounded to 4 decimal places
            if isinstance(v, float):
                logger.info(f"{k}: {v:.4f}")
            else:
                logger.info(f"{k}: {v}")

        for k, v in model.get_metrics_dict(scores).items():
            results["test_" + k] = v

        # Save to JSON file with indentation
        converted_results = convert_np_float_to_float(results)
        with open(f"{output_dir}/baselineRSNN-results-{session_name}.json", "w") as f:
            json.dump(converted_results, f, indent=4)

        if cfg.plotting.plot_training:
            fig, ax = plot_training(
                results,
                cfg.training.nb_epochs_train,
                names=["loss", "r2"],
                save_path=output_dir / f"baselineRSNN_training_{session_name}.png"
            )

            
            
if __name__ == "__main__":
    train_rsnn_tiny()