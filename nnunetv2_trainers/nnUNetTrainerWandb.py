"""
nnU-Net Trainer with Weights & Biases (W&B) logging.

Supports offline mode for HPC clusters without internet access.
Sync runs later with: wandb sync <run_dir>
"""

import os
from typing import List

import numpy as np

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerWandb(nnUNetTrainer):
    """
    nnUNetTrainer with Weights & Biases logging support.

    Environment variables:
        WANDB_MODE: Set to 'offline' for clusters without internet (default: offline)
        WANDB_PROJECT: Project name (default: nnunet)
        WANDB_ENTITY: Team/user name (optional)
        WANDB_RUN_NAME: Custom run name (optional, auto-generated if not set)

    Usage:
        nnUNetv2_train DATASET CONFIG FOLD -tr nnUNetTrainerWandb
    """

    def __init__(self, plans, configuration, fold, dataset_json, device):
        super().__init__(plans, configuration, fold, dataset_json, device)

        self.wandb_initialized = False
        self.wandb_run = None

        # Set offline mode by default for HPC
        if 'WANDB_MODE' not in os.environ:
            os.environ['WANDB_MODE'] = 'offline'

    def initialize(self):
        """Initialize trainer and W&B."""
        super().initialize()

        if self.local_rank == 0:
            self._init_wandb()

    def _init_wandb(self):
        """Initialize W&B run."""
        try:
            import wandb

            # Get config from environment or use defaults
            project = os.environ.get('WANDB_PROJECT', 'nnunet-lumbar-muscle')
            entity = os.environ.get('WANDB_ENTITY', None)
            run_name = os.environ.get('WANDB_RUN_NAME', None)

            # Auto-generate run name if not provided
            if run_name is None:
                dataset_name = self.plans_manager.dataset_name
                run_name = f"{dataset_name}_{self.configuration_name}_fold{self.fold}"

            # Build config dict
            config = {
                'dataset': self.plans_manager.dataset_name,
                'configuration': self.configuration_name,
                'fold': self.fold,
                'num_epochs': self.num_epochs,
                'batch_size': self.configuration_manager.batch_size,
                'patch_size': list(self.configuration_manager.patch_size),
                'num_classes': self.label_manager.num_segmentation_heads,
                'learning_rate': self.initial_lr,
            }

            # Initialize W&B
            self.wandb_run = wandb.init(
                project=project,
                entity=entity,
                name=run_name,
                config=config,
                dir=self.output_folder,
                resume='allow',
            )

            self.wandb_initialized = True
            self.print_to_log_file(f"W&B initialized: {wandb.run.dir}")
            self.print_to_log_file(f"W&B mode: {os.environ.get('WANDB_MODE', 'online')}")

        except Exception as e:
            self.print_to_log_file(f"W&B initialization failed: {e}")
            self.print_to_log_file("Continuing without W&B logging")
            self.wandb_initialized = False

    def on_train_epoch_end(self, train_outputs: List[dict]):
        """Log training metrics to W&B."""
        super().on_train_epoch_end(train_outputs)

        if self.wandb_initialized and self.local_rank == 0:
            try:
                import wandb

                train_loss = self.logger.my_fantastic_logging['train_losses'][-1]
                lr = self.optimizer.param_groups[0]['lr']

                wandb.log({
                    'train/loss': train_loss,
                    'train/learning_rate': lr,
                    'epoch': self.current_epoch,
                }, step=self.current_epoch)

            except Exception as e:
                self.print_to_log_file(f"W&B train logging error: {e}")

    def on_validation_epoch_end(self, val_outputs: List[dict]):
        """Log validation metrics to W&B."""
        super().on_validation_epoch_end(val_outputs)

        if self.wandb_initialized and self.local_rank == 0:
            try:
                import wandb

                val_loss = self.logger.my_fantastic_logging['val_losses'][-1]
                mean_fg_dice = self.logger.my_fantastic_logging['mean_fg_dice'][-1]
                ema_fg_dice = self.logger.my_fantastic_logging['ema_fg_dice'][-1]
                dice_per_class = self.logger.my_fantastic_logging['dice_per_class_or_region'][-1]

                log_dict = {
                    'val/loss': val_loss,
                    'val/mean_fg_dice': mean_fg_dice,
                    'val/ema_fg_dice': ema_fg_dice,
                    'epoch': self.current_epoch,
                }

                # Log per-class dice scores
                if isinstance(dice_per_class, (list, np.ndarray)):
                    class_names = ['L_ES', 'R_ES', 'L_MF', 'R_MF']
                    for i, dice in enumerate(dice_per_class):
                        if i < len(class_names):
                            log_dict[f'val/dice_{class_names[i]}'] = dice
                        else:
                            log_dict[f'val/dice_class_{i}'] = dice

                wandb.log(log_dict, step=self.current_epoch)

            except Exception as e:
                self.print_to_log_file(f"W&B val logging error: {e}")

    def on_train_end(self):
        """Finish W&B run."""
        super().on_train_end()

        if self.wandb_initialized and self.local_rank == 0:
            try:
                import wandb

                # Log final best metrics
                if hasattr(self, '_best_ema') and self._best_ema is not None:
                    wandb.run.summary['best_ema_fg_dice'] = self._best_ema

                wandb.finish()
                self.print_to_log_file("W&B run finished")
                self.print_to_log_file(f"To sync offline run: wandb sync {self.output_folder}/wandb/latest-run")

            except Exception as e:
                self.print_to_log_file(f"W&B finish error: {e}")
