import numpy as np
import random
import copy
import matplotlib.pyplot as plt

from stork.generators import StandardGenerator

from .loss import get_train_loss
from .optimizer import get_optimizer
from .scheduler import get_lr_scheduler
from .pruning import prune_model

from efficient_rsnn_bmi.utils.plotting import plot_activity_snapshot

def configure_model(model, cfg, train_data, dtype):

    loss_stack = get_train_loss(cfg, train_data)
    opt, opt_kwargs = get_optimizer(cfg, dtype)
    
    if cfg.training.lr_scheduler is not None:
        scheduler, scheduler_kwargs = get_lr_scheduler(cfg, opt)
    else:
        scheduler = None
        scheduler_kwargs = None
        
    def worker_init_fn(worker_id):
        np.random.seed(cfg.seed + worker_id)
        random.seed(cfg.seed + worker_id)
    
    generator = StandardGenerator(nb_workers=cfg.nb_workers, worker_init_fn=worker_init_fn)

    # Configure model
    model.configure(
        input=model.groups[0],
        output=model.groups[-1],
        loss_stack=loss_stack,
        generator=generator,
        optimizer=opt,
        optimizer_kwargs=opt_kwargs,
        scheduler=scheduler,
        scheduler_kwargs=scheduler_kwargs,
        time_step=cfg.datasets.dt,
    )

    return model

def train_validate_model(
    model,
    cfg,
    train_data,
    valid_data,
    nb_epochs,
    verbose=True,
    snapshot_prefix="",
):

    if cfg.plotting.plot_snapshots:
        fix, ax = plot_activity_snapshot(
            model, valid_data, save_path=snapshot_prefix + "snapshot_before.png"
        )

    history = model.fit_validate(
        train_data,
        valid_data,
        nb_epochs=nb_epochs,
        verbose=verbose,
    )

    if cfg.plotting.plot_snapshots:
        fig, ax = plot_activity_snapshot(
            model,
            valid_data,
            save_path=snapshot_prefix + "snapshot_after_e{}.png".format(nb_epochs),
        )

    return model, history

def prune_retrain_model_iterate(
        ori_model,
        cfg,
        train_data,
        valid_data,
        logger,
        r2_train_before_pruned,
        r2_val_before_pruned,
        nb_epochs_retrain=50,
        prune_percentage_start=0.1,
        tolerance=0.03, 
        prune_precision=[0.1], # go through one by one, should decrease gradually, for example, [0.1, 0.01]
        max_prune_percentage=1.0,
        is_plot_pruning=True,
        is_pruning_ver=False,
        session_name='None',
        pruning_plot_prefix=''
):

    sparsity_scores = []
    r2_trains = []
    r2_vals = []
    prune_percentages = []
    precision_idx = 0
    logger.info(f"r2 performance of unpruned model: training: {r2_train_before_pruned:.4f}, validation: {r2_val_before_pruned:.4f}")
    logger.info("Pruning model iteratively...")
    prune_percentage = prune_percentage_start
    ori_state_dict = copy.deepcopy(ori_model.state_dict())
    final_state_dict = copy.deepcopy(ori_model.state_dict()) # to be updated in the loop, unless pruning can't work at all
    pruned_model = ori_model # Note: just rename

    while prune_percentage <= max_prune_percentage:
        pruned_model.load_state_dict(ori_state_dict)

        pruned_model, r2_train_after_retraining, r2_val_after_retraining, sparsity_after_retraining = prune_retrain_model(
                                                                                                                pruned_model,
                                                                                                                prune_percentage,
                                                                                                                cfg,
                                                                                                                train_data,
                                                                                                                valid_data,
                                                                                                                logger,
                                                                                                                nb_epochs_retrain,
                                                                                                                is_pruning_ver,
                                                                                                                session_name)


        if r2_train_after_retraining-r2_train_before_pruned < -np.abs(tolerance*r2_train_before_pruned): # or r2_val_after_retraining-r2_val_before_pruned < -np.abs(tolerance*r2_val_before_pruned):
            if precision_idx < len(prune_precision)-1:
                prune_percentage -= prune_precision[precision_idx]
                precision_idx += 1
                prune_percentage += prune_precision[precision_idx]
            else:
                break
            continue

        r2_trains.append(r2_train_after_retraining)
        r2_vals.append(r2_val_after_retraining)
        sparsity_scores.append(sparsity_after_retraining)
        logger.info(f"Pruning percentage: {prune_percentage:.2f}, Sparsity: {sparsity_after_retraining:.4f}, r2 after retraining: training: {r2_train_after_retraining:.4f}, validation: {r2_val_after_retraining:.4f}")
        prune_percentages.append(prune_percentage)

        final_state_dict = copy.deepcopy(pruned_model.state_dict())
        prune_percentage += prune_precision[precision_idx]

    if is_plot_pruning: 

        # Plot the results
        fig = plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.scatter(prune_percentages, r2_trains, marker='o', label='trained_pruned')
        plt.scatter(prune_percentages, r2_vals, marker='o', label='val_pruned')
        plt.axhline(y=r2_train_before_pruned, color='b', linestyle='--', label='trained_Unpruned')
        plt.axhline(y=r2_val_before_pruned, color='r', linestyle='--', label='val_Unpruned')
        plt.xlabel("Pruning Percentage")
        plt.ylabel("r2_mean")
        plt.title("Impact of Pruning and Retraining on Regression Performance")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.scatter(prune_percentages, sparsity_scores, marker='o')
        plt.xlabel("Pruning Percentage")
        plt.ylabel("Sparsity")
        plt.title("Connection Sparsity vs. Pruning Percentage")

        fig.savefig(pruning_plot_prefix+' Pruning.png', dpi=250)

    if prune_percentages:
        logger.info(f"Maximum pruning percentage that retains the performance of the unpruned model: {prune_percentages[-1]:.2f}")
    else:
        logger.info(f"No suitable pruned model found, try smaller starting pruning percentage.")

    pruned_model.load_state_dict(final_state_dict)

    return pruned_model

def prune_retrain_model(
        model,
        prune_percentage,
        cfg,
        train_data,
        valid_data,
        logger,
        nb_epochs_retrain=50,
        is_pruning_ver=False,
        session_name='None',
):
    # Calculate the connection sparsity
    def calculate_con_sparsity(model):
        total_params = 0
        zero_params = 0
        for name, param in model.named_parameters():
            if 'weight' in name:
                total_params += param.numel()
                zero_params += (param == 0).sum().item()
        return zero_params / total_params
    
    mask = prune_model(model, prune_percentage)
    logger.info(f"Pruning percentage: {prune_percentage:.2f}")
    logger.info("Retrain the pruned model while keeping the pruned weights as zero...")
    
    # Re-configure optimizer and learning rate scheduler
    model.configure_optimizer(model.optimizer_class, model.optimizer_kwargs)
    new_scheduler_kwargs = {"T_max": nb_epochs_retrain}
    model.configure_scheduler(model.scheduler_class, new_scheduler_kwargs)
    
    model, history = train_validate_model_pruning(
        model,
        cfg,
        train_data,
        valid_data,
        nb_epochs_retrain,
        verbose=is_pruning_ver, # cfg.training.verbose,
        snapshot_prefix=session_name + ' Pruned ' + str(np.round(prune_percentage,2)) + "_",
        mask=mask,
    )
    r2_train_after_retraining = history['r2'][-1]
    r2_val_after_retraining = history['val_r2'][-1]
    sparsity_after_retraining = calculate_con_sparsity(model)

    return model, r2_train_after_retraining, r2_val_after_retraining, sparsity_after_retraining

def train_validate_model_pruning(
    model,
    cfg,
    train_data,
    valid_data,
    nb_epochs,
    verbose=True,
    snapshot_prefix="",
    mask=None,
    ):

    if cfg.plotting.plot_snapshots:
        fix, ax = plot_activity_snapshot(
            model, valid_data, save_path=snapshot_prefix + "snapshot_before.png"
        )
    
    history = model.fit_validate_masked(
        train_data,
        valid_data,
        nb_epochs=nb_epochs,
        verbose=verbose,
        mask=mask,
    )

    if cfg.plotting.plot_snapshots:
        fig, ax = plot_activity_snapshot(
            model,
            valid_data,
            save_path=snapshot_prefix + "snapshot_after_e{}.png".format(nb_epochs),
        )

    return model, history


def prune_retrain_model(
        model,
        prune_percentage,
        cfg,
        train_data,
        valid_data,
        logger,
        nb_epochs_retrain=50,
        is_pruning_ver=False,
        session_name='None',
):
    # Calculate the connection sparsity
    def calculate_con_sparsity(model):
        total_params = 0
        zero_params = 0
        for name, param in model.named_parameters():
            if 'weight' in name:
                total_params += param.numel()
                zero_params += (param == 0).sum().item()
        return zero_params / total_params
    
    mask = prune_model(model, prune_percentage)
    logger.info(f"Pruning percentage: {prune_percentage:.2f}")
    logger.info("Retrain the pruned model while keeping the pruned weights as zero...")
    
    # Re-configure optimizer and learning rate scheduler
    model.configure_optimizer(model.optimizer_class, model.optimizer_kwargs)
    new_scheduler_kwargs = {"T_max": nb_epochs_retrain}
    model.configure_scheduler(model.scheduler_class, new_scheduler_kwargs)
    
    model, history = train_validate_model_pruning(
        model,
        cfg,
        train_data,
        valid_data,
        nb_epochs_retrain,
        verbose=is_pruning_ver, # cfg.training.verbose,
        snapshot_prefix=session_name + ' Pruned ' + str(np.round(prune_percentage,2)) + "_",
        mask=mask,
    )
    r2_train_after_retraining = history['r2'][-1]
    r2_val_after_retraining = history['val_r2'][-1]
    sparsity_after_retraining = calculate_con_sparsity(model)

    return model, r2_train_after_retraining, r2_val_after_retraining, sparsity_after_retraining