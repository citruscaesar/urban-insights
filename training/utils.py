import torchmetrics

from pathlib import Path
from torch import optim, nn
from typing import Any, Optional
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from etl.etl import validate_dir
from typing import Literal

METRICS = {
    "accuracy": torchmetrics.Accuracy,
    "f1": torchmetrics.F1Score,
    "iou": torchmetrics.JaccardIndex,
    "confusion_matrix": torchmetrics.ConfusionMatrix,
    "cohen_kappa": torchmetrics.CohenKappa,
    "auroc": torchmetrics.AUROC,
}

LOSSES = {
    "binary_cross_entropy": nn.BCEWithLogitsLoss,
    "cross_entropy": nn.CrossEntropyLoss,
    "mean_squared_error": nn.MSELoss,
}

OPTIMIZERS = {
    "sgd": optim.SGD,
    "adam": optim.Adam,
}

def Loss(loss_name: str, loss_kwargs: Optional[dict[str, Any]] = None):
    assert loss_name in LOSSES, f"{loss_name} is not implemented"
    return LOSSES[loss_name](**loss_kwargs)

def Metric(metric_name: str, metric_kwargs: dict[str, Any]) -> torchmetrics.Metric:
    assert metric_name in METRICS, f"{metric_name} is not implemented"
    return METRICS[metric_name](**metric_kwargs)

def Optimizer(optimizer_name: str, optimizer_kwargs: dict[str, Any]):
    assert optimizer_name in OPTIMIZERS, f"{optimizer_name} is not implemented"
    return OPTIMIZERS[optimizer_name](**optimizer_kwargs)

def setup_logger(logs_dir: Path, name: str | int, log_freq: int = 100):
    logger = CSVLogger(
       save_dir=logs_dir.parent,
       name=logs_dir.name,
       version=name,
       flush_logs_every_n_steps=log_freq,
    )
    print(f"Local Logging To : {logger.log_dir}")
    return logger

def setup_wandb_logger(logs_dir: Path, name: str | int, log_freq: int = 100):
    assert name is not None, "experiment name not provided"
    save_dir = validate_dir(logs_dir, str(name))
    logger = WandbLogger(
        project = logs_dir.name,
        name = str(name),
        save_dir = save_dir,
        log_model = True,
        resume = "auto",
        save_code = True,
    )
    print(f"WandB Logging To: {save_dir/'wandb'}")
    return logger

def setup_checkpoint(ckpt_dir: Path, metric: str, mode: Literal["min", "max"], save_top_k: int | Literal["all"], **kwargs) -> ModelCheckpoint:
    monitor_metric = f"val/{metric}";
    callback = ModelCheckpoint(
        dirpath = ckpt_dir,
        monitor = monitor_metric,
        mode = mode,
        filename = f"{{epoch}}_{{step}}",
        save_top_k = -1 if isinstance(save_top_k, str) else save_top_k,
        save_last = True,
        save_on_train_epoch_end = False,
    )
    print(f"Checkpoint Monitoring: {monitor_metric}, Checkpoints Saved To: {ckpt_dir}")
    return callback