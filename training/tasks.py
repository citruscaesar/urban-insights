# General Purpose Libraries
import torch
from torchmetrics import MetricCollection

# Metrics, Criterion and Optimizers
from training.utils import Loss, Metric, Optimizer

# Lightning Module
from lightning import LightningModule

from typing import Any, Optional, Literal
from torch import Tensor

class ClassificationTask(LightningModule):
    def __init__(
            self, 
            model: torch.nn.Module,
            task: Literal["classification", "segmentation"],
            model_name: str,
            model_params: dict, 
            loss: str,
            loss_params: dict[str, Any],
            optimizer: str,
            optimizer_params: dict[str, Any],
            monitor_metric: str,
            num_classes: int,
            batch_size: int,
            grad_accum: int,
            class_names: tuple[str],
            **kwargs
        ) -> None:

        assert task in ("classification", "segmentation"), f"task = {task} is invalid"

        super().__init__()
        self.model = model
        self.task = task
        self.num_classes = num_classes
        self.class_names = class_names
        self.batch_size = batch_size // grad_accum
        self.criterion = Loss(loss, loss_params)
        self.optimizer_name = optimizer
        self.optimizer_params = optimizer_params
        self.__set_metrics(monitor_metric)
        self.save_hyperparameters(
            "model_name", "model_params", "loss", "loss_params", 
            "optimizer", "optimizer_params", "monitor_metric")
    
    def _forward(self, batch) -> tuple[Tensor, Tensor, Tensor]:
        # NOTE: masks(NCHW).argmax(1) converts one_hot back to categorical, same with preds
        # print(f"Image Batch: {images.shape}, Mask Batch: {masks.shape}", end = ' ')
        # print(f"Preds Batch: {preds.shape}", end = ' ')
        # print(f"Loss: {loss}")
        images, masks = batch[0], batch[1]
        preds = self.model(images)
        loss = self.criterion(preds, masks)
        return preds, masks, loss

    def forward(self, batch) -> Any:
        images = batch[0]
        images.requires_grad = True
        print(images.shape, images.dtype, images.device, images.min(), images.max())
        return self.model(images)
    
    def training_step(self, batch, batch_idx):
        preds, masks, loss = self._forward(batch) 
        self.train_metrics.update(preds, masks)

        # NOTE: "train/loss" must have on_step = True and on_epoch = True
        self.log(f"train/loss", loss, on_epoch = True, on_step = True, batch_size = self.batch_size);
        self.log("lr", self.optimizer_params["lr"], on_step = False, on_epoch = True)
        self.log_dict(self.train_metrics, on_epoch = True, batch_size = self.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        preds, labels, loss = self._forward(batch) 
        self.val_losses.append(loss)
        self.val_metrics.update(preds, labels)
        self.val_confusion_matrix.update(preds, labels)
        return preds 

    def on_validation_epoch_end(self):
        self.log(f"val/loss", torch.tensor(self.val_losses).mean());
        self.log_dict(self.val_metrics.compute())
        self.val_losses.clear()
        self.val_metrics.reset()
        self.val_confusion_matrix.reset()

    def test_step(self, batch, batch_idx):
        preds, labels, loss = self._forward(batch) 
        self.test_losses.append(loss)
        self.test_metrics.update(preds, labels)
        self.test_confusion_matrix.update(preds, labels)
        return preds 

    def on_test_epoch_end(self):
        self.log(f"test/loss", torch.tensor(self.test_losses).mean());
        self.log_dict(self.test_metrics.compute())
        self.test_losses.clear()
        self.test_metrics.reset()
        self.test_confusion_matrix.reset()
    
    def configure_optimizers(self):
        _optimizer_params = self.optimizer_params.copy()
        _optimizer_params["params"] = self.model.parameters()
        return Optimizer(self.optimizer_name, _optimizer_params)

    def __set_metrics(self, monitor_metric: str):
        # print(f"monitor metric: {monitor_metric}")
        metric_params = {
            "task" : "multiclass" if self.num_classes > 2 else "binary",
            "num_classes": self.num_classes,
        }

        metrics = MetricCollection({
            # NOTE: add additional metrics here, eg. 
            # "cohen_kappa": Metric("cohen_kappa", metric_params),
            monitor_metric: Metric(monitor_metric, metric_params),
        })

        self.train_metrics = metrics.clone(prefix = "train/")
        self.val_metrics = metrics.clone(prefix = "val/")
        self.test_metrics = metrics.clone(prefix = "test/")

        self.val_losses = list()
        self.test_losses = list()
        self.val_confusion_matrix = Metric("confusion_matrix", metric_params)
        self.test_confusion_matrix = Metric("confusion_matrix", metric_params)