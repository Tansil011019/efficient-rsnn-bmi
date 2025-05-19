# Evaluate all models in the 'models' directory
# Results are saved to results_bigRSNN.json and results_tinyRSNN.json

# CONFIG
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import to_absolute_path
from pathlib import Path

# BENCHMARKING / NUMERIC
import torch
from torch.utils.data import DataLoader, Subset
from neurobench.datasets import PrimateReaching
from neurobench.benchmarks import Benchmark
from neurobench.models import TorchModel
import snntorch as snn

# LOCAL
# from challenge.utils.snntorch import get_bigRSNN_model, get_tinyRSNN_model
# from challenge.evaluate import benchmark_model
# from challenge.train import configure_model
# from challenge.model import get_model
# from challenge.data import get_dataloader

from efficient_rsnn_bmi.core.dataloader import get_dataloader
from efficient_rsnn_bmi.core.model import get_model
from efficient_rsnn_bmi.core.train import configure_model, train_validate_model, prune_retrain_model_iterate
from efficient_rsnn_bmi.core.evaluate import evaluate_model, benchmark_model

# LOGGING
import logging
import json

# SET UP LOGGER
logging.basicConfig()
logger = logging.getLogger("train-bigRSNN")

# Results handler
# # # # # # # # # #


class BenchmarkResultsHandler:
    def __init__(self):
        self.footprint = []
        self.connection_sparsity = []
        self.activation_sparsity = []
        self.dense = []
        self.macs = []
        self.acs = []
        self.r2 = []

    def append(self, results):
        self.footprint.append(results["footprint"])
        self.connection_sparsity.append(results["connection_sparsity"])
        self.activation_sparsity.append(results["activation_sparsity"])
        self.dense.append(results["synaptic_operations"]["Dense"])
        self.macs.append(results["synaptic_operations"]["Effective_MACs"])
        self.acs.append(results["synaptic_operations"]["Effective_ACs"])
        self.r2.append(results["r2"])

    def get_summary(self):
        return {
            "footprint": sum(self.footprint) / len(self.footprint),
            "connection_sparsity": sum(self.connection_sparsity)
            / len(self.connection_sparsity),
            "activation_sparsity": sum(self.activation_sparsity)
            / len(self.activation_sparsity),
            "dense": sum(self.dense) / len(self.dense),
            "macs": sum(self.macs) / len(self.macs),
            "acs": sum(self.acs) / len(self.acs),
            "r2": sum(self.r2) / len(self.r2),
        }

    def to_dict(self):
        return {
            "footprint": self.footprint,
            "connection_sparsity": self.connection_sparsity,
            "activation_sparsity": self.activation_sparsity,
            "dense": self.dense,
            "macs": self.macs,
            "acs": self.acs,
            "r2": self.r2,
        }


# Functions for stork models
# # # # # # # # # #


def load_stork_model(cfg, state_dict_dir, train_dat, monkey_name, dtype):
    logger.info("Initializing model...")
    model = get_model(
        cfg, nb_inputs=cfg.datasets.nb_inputs[monkey_name], dtype=dtype, data=train_dat
    )
    model = configure_model(model, cfg, train_dat, dtype)

    logger.info("Loading model state from {}".format(state_dict_dir))
    model.load_state_dict(torch.load(state_dict_dir))

    return model


def load_and_benchmark_stork_model(
    cfg, modelname, state_dict_dir, train_dat, test_dat, monkey_name, dtype
):
    """
    Load and benchmark a stork model
    """

    # Make copy of config so we can add a model field
    # with the correct model configuration
    this_cfg = OmegaConf.structured(OmegaConf.to_yaml(cfg))

    if modelname == "bigRSNN":
        this_cfg.model = cfg.bigRSNN
        this_cfg.training = cfg.bigRSNNtraining
    elif modelname == "baselineRSNN":
        this_cfg.model = cfg.baselineRSNN
        this_cfg.training = cfg.baselineRSNNtraining
    else:
        raise ValueError("Invalid modelname. Must be 'bigRSNN' or 'baseline-rsnn'")

    model = load_stork_model(this_cfg, state_dict_dir, train_dat, monkey_name, dtype)

    if modelname == "baselineRSNN":
        logger.info("Converting model to half precision...")
        model = model.half()
        test_dat.dtype = torch.float16

    bm_results = benchmark_model(model, this_cfg, test_dat)

    return model, bm_results


def benchmark_all_models_stork(
    cfg, modelname, state_dict_paths, train_dat, test_dat, monkey_name, dtype
):
    """
    Benchmark all models in the given state_dict_paths
    """

    # Prepare results dictionary
    results = BenchmarkResultsHandler()

    for idx, state_dict_path in enumerate(state_dict_paths):
        logger.info("=" * 50)
        logger.info("Evaluating model {} of {}".format(idx + 1, len(state_dict_paths)))

        # Load and benchmark the model
        _, bm_results = load_and_benchmark_stork_model(
            cfg, modelname, state_dict_path, train_dat, test_dat, monkey_name, dtype
        )

        # Log summary of results
        logger.info("Benchmark results:")
        logger.info("R2: {}".format(bm_results["R2"]))
        logger.info("Footprint: {}".format(bm_results["Footprint"]))
        logger.info("Connection Sparsity: {}".format(bm_results["'ConnectionSparsity"]))
        logger.info("Activation Sparsity: {}".format(bm_results["ActivationSparsity"]))
        logger.info("Dense: {}".format(bm_results["SynapticOperations"]["Dense"]))
        logger.info(
            "MACs: {}".format(bm_results["SynapticOperations"]["Effective_MACs"])
        )
        logger.info(
            "ACs: {}".format(bm_results["SynapticOperations"]["Effective_ACs"])
        )

        # Save results
        results.append(bm_results)

    return results


# Functions for snnTorch models
# # # # # # # # # #


# def load_data_snnTorch(cfg, filename, dtype):
#     # Load the dataset
#     dataset = PrimateReaching(
#         file_path=cfg.datasets.data_dir,
#         filename=filename,
#         num_steps=1,
#         train_ratio=0.5,
#         bin_width=4e-3,
#         biological_delay=0,
#         remove_segments_inactive=False,
#     )

#     test_set_loader = DataLoader(
#         Subset(dataset, dataset.ind_test),
#         batch_size=len(dataset.ind_test),
#         shuffle=False,
#     )

#     return dataset, test_set_loader


# def load_snnTorch_model(cfg, modelname, state_dict_dir, device):
#     if modelname == "bigRSNN":
#         logger.info("Creating bigRSNN snnTorch model...")
#         model = get_bigRSNN_model(cfg.bigRSNN, torch.load(state_dict_dir), device)

#     elif modelname == "tinyRSNN":
#         logger.info("Creating tinyRSNN snnTorch model...")
#         model = get_tinyRSNN_model(cfg.tinyRSNN, torch.load(state_dict_dir), device)

#     return model


# def load_and_benchmark_snnTorch_model(
#     cfg, modelname, state_dict_path, test_set_loader, dtype
# ):
#     model = load_snnTorch_model(cfg, modelname, state_dict_path, cfg.device)

#     # Benchmark using NeuroBench
#     model.reset()
#     BMmodel = TorchModel(model)
#     BMmodel.add_activation_module(snn.SpikingNeuron)

#     benchmark = Benchmark(
#         BMmodel, test_set_loader, [], [], [cfg.static_metrics, cfg.workload_metrics]
#     )

#     results = benchmark.run(device=cfg.device)

#     return model, results


# def benchmark_all_models_snnTorch(
#     cfg, modelname, state_dict_paths, test_set_loader, dtype
# ):
#     """
#     Benchmark all models in the given state_dict_paths
#     """

#     # Prepare results dictionary
#     results = BenchmarkResultsHandler()

#     for idx, state_dict_path in enumerate(state_dict_paths):
#         logger.info("=" * 50)
#         logger.info("Evaluating model {} of {}".format(idx + 1, len(state_dict_paths)))

#         # Load and benchmark the model
#         _, bm_results = load_and_benchmark_snnTorch_model(
#             cfg, modelname, state_dict_path, test_set_loader, dtype
#         )

#         # Log summary of results
#         logger.info("Benchmark results:")
#         logger.info("R2: {}".format(bm_results["r2"]))
#         logger.info("Footprint: {}".format(bm_results["footprint"]))
#         logger.info("Connection Sparsity: {}".format(bm_results["connection_sparsity"]))
#         logger.info("Activation Sparsity: {}".format(bm_results["activation_sparsity"]))
#         logger.info("Dense: {}".format(bm_results["synaptic_operations"]["Dense"]))
#         logger.info(
#             "MACs: {}".format(bm_results["synaptic_operations"]["Effective_MACs"])
#         )
#         logger.info(
#             "ACs: {}".format(bm_results["synaptic_operations"]["Effective_ACs"])
#         )

#         # Save results
#         results.append(bm_results)

#     return results


@hydra.main(config_path="../config", config_name="evaluate", version_base=None)
def evaluate(cfg: DictConfig) -> None:
    logger.info(
        "Evaluating {} models in the {} directory".format(cfg.modelname, cfg.model_dir)
    )

    # # # # # # # # #
    # SETUP
    # # # # # # # # #

    # dtype
    dtype = getattr(torch, cfg.dtype)

    # Model for evaluation
    if cfg.modelname == "all":
        cfg.modelname = ["bigRSNN", "baselineRSNN"]
    elif cfg.modelname in ["bigRSNN", "baselineRSNN"]:
        cfg.modelname = [cfg.modelname]
    else:
        raise ValueError("Invalid model type. Must be 'bigRSNN', 'baselineRSNN', or 'all'")

    # Data loader for stork models
    storkmodel_dataloader = get_dataloader(cfg, dtype=dtype)

    # Results dictionary
    results = {key: {} for key in cfg.modelname}

    # ITERATE OVER SESSIONS
    # # # # # # # # # # # # #
    all_session_filenames = []

    # print(cfg.datasets.filenames)

    for monkey_name in ["loco", "indy"]:
        for session, filename in cfg.datasets.filenames[monkey_name].items():
            logger.info("=" * 50)
            logger.info("Evaluating session {} ({})".format(session, filename))
            logger.info("=" * 50)

            all_session_filenames.append(filename)

            for model_type in cfg.modelname:
                logger.info("Evaluating {} models...".format(model_type))

                # Find models
                model_dir = Path(to_absolute_path(cfg.model_dir)) / session
                state_dict_paths = [
                    Path(f) for f in model_dir.iterdir() if model_type in f.name
                ]
                logger.info(
                    "Found {} {} models for session {}".format(
                        len(state_dict_paths), model_type, session
                    )
                )

                if cfg.use_snnTorch_model:
                    # logger.info("Using snnTorch models...")

                    # dataset, test_set_loader = load_data_snnTorch(cfg, filename, dtype)
                    # session_results = benchmark_all_models_snnTorch(
                    #     cfg, model_type, state_dict_paths, test_set_loader, dtype
                    # )
                    pass

                else:
                    logger.info("Using stork models...")
                    train_dat, _, test_dat = (
                        storkmodel_dataloader.get_single_session_data(filename)
                    )
                    session_results = benchmark_all_models_stork(
                        cfg,
                        model_type,
                        state_dict_paths,
                        train_dat,
                        test_dat,
                        monkey_name,
                        dtype,
                    )

                # Log summary of results
                session_results_summary = session_results.get_summary()
                logger.info("Summary of results:")
                for k, v in session_results_summary.items():
                    logger.info("{}: {}".format(k, v))

                # Save results from individual seeds to a json file
                session_results_save_path = Path(to_absolute_path("results"))
                with open(
                    session_results_save_path
                    / "results_{}_{}.json".format(model_type, filename),
                    "w",
                ) as f:
                    json.dump(session_results.to_dict(), f, indent=4)
                logger.info(
                    "Saved all session results to results_{}_{}.json".format(
                        model_type, filename
                    )
                )

                # Save results summary
                results[model_type][filename] = session_results_summary

    logger.info("=" * 50)
    logger.info("=" * 50)
    logger.info("Evaluation complete.")

    # Save results by model type
    for model_type in cfg.modelname:
        # Compute average across all sessions
        results[model_type]["average"] = {}
        for metric in [
            "footprint",
            "connection_sparsity",
            "activation_sparsity",
            "dense",
            "macs",
            "acs",
            "r2",
        ]:
            results[model_type]["average"][metric] = sum(
                [results[model_type][file][metric] for file in all_session_filenames]
            ) / len(all_session_filenames)

        # Log average results
        logger.info("=" * 50)
        logger.info("Average results for {} models:".format(model_type))
        for k, v in results[model_type]["average"].items():
            logger.info("{}: {}".format(k, v))
        logger.info("=" * 50)

        # Save each 'results-{model_type}' to a json file
        # To the root directory
        result_save_path = Path(to_absolute_path(""))
        with open(
            result_save_path / "results_summary_{}.json".format(model_type), "w"
        ) as f:
            json.dump(results[model_type], f, indent=4)

        logger.info("Saved results to results_summary_{}.json".format(model_type))


if __name__ == "__main__":
    evaluate()