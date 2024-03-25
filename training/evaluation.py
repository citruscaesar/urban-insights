from pathlib import Path
import torch
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from lightning import Trainer, LightningModule 
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from lightning.pytorch.callbacks import Callback
from torchmetrics import ConfusionMatrix
from torchmetrics.functional import jaccard_index, f1_score

from etl.etl import validate_dir

from typing import Any, Optional, Literal
from numpy.typing import NDArray

# TODO: Find a light and quick to install tex
# plt.rcParams["text.usetex"] = True
plt.rcParams["axes.grid"] = True

class EvaluationReport:
    @classmethod
    def get_metrics_dict(cls, confusion_matrix: NDArray, prefix:str, class_names: Optional[tuple[str, ...]] = None) -> dict[str, float]:
        df = cls.get_metrics_df(confusion_matrix, class_names)
        metrics = dict()
        for class_name, row in df.iterrows(): # type: ignore
            if str(class_name) == "accuracy":
                metrics[prefix + "accuracy"] = row["precision"]
                continue
            class_name = str(class_name).replace(' ', '_') + '_'
            metrics[prefix + class_name + "precision"] = row["precision"]
            metrics[prefix + class_name + "recall"] = row["recall"]
            metrics[prefix + class_name + "iou"] = row["iou"]
            metrics[prefix + class_name + "f1"] = row["f1"]
            metrics[prefix + class_name + "support"] = int(row["support"])
        return metrics

    @staticmethod
    def get_metrics_df(confusion_matrix: NDArray, class_names: Optional[tuple[str, ...]] = None) -> pd.DataFrame:
        num_classes = confusion_matrix.shape[0]
        num_samples = np.sum(confusion_matrix)

        if isinstance(class_names, tuple):
            assert len(class_names) == num_classes, f"invalid shape, expected len(class_names) = {num_classes}, received = {len(class_names)},"
        else:
            class_names = tuple(str(c) for c in range(num_classes))

        # NOTE: If required, add additional metrics BEFORE the support column
        df = pd.DataFrame(columns = ["class_name", "precision", "recall", "iou", "f1", "support"])
        for c in range(num_classes):
            tp = confusion_matrix[c, c]
            p_hat = np.sum(confusion_matrix[:, c])
            p = np.sum(confusion_matrix[c, :])

            precision = (tp / p_hat) if p_hat > 0 else 0
            recall = (tp / p) if p > 0 else 0
            iou = tp / (p+p_hat-tp) if (p+p_hat-tp) > 0 else 0
            f1 =  (2*tp) / (p+p_hat) if (p+p_hat) > 0 else 0
            support = np.sum(p)

            df.loc[len(df.index)] = [class_names[c].lower(), precision, recall, iou, f1, support]
            
        accuracy = confusion_matrix.trace() / num_samples

        # NOTE weighted_metric = np.dot(metric, support) 
        weighted_metrics = np.matmul((df["support"] / df["support"].sum()).to_numpy(), 
                                    df[["precision", "recall", "iou", "f1"]].to_numpy())

        df.loc[len(df.index)] = ["accuracy", accuracy, accuracy, accuracy, accuracy, num_samples]
        df.loc[len(df.index)] = ["macro", df["precision"].mean(), df["recall"].mean(), df["iou"].mean(), df["f1"].mean(), num_samples]
        df.loc[len(df.index)] = ["weighted", *weighted_metrics, num_samples]
        df.set_index("class_name", inplace = True)
        return df

    @staticmethod
    def get_logs_df(logs_dir: Path, monitor_metric: str):
        "returns dataframe with loss values and monitored metric values logged during training and evaluation"

        def filter_columns(df: pd.DataFrame, metric: str) -> pd.DataFrame:
            cols = ["epoch", "step", "train/loss_step", "train/loss_epoch", f"train/{metric}"]
            df[f"train/{metric}"] = np.nan
            if f"train/{metric}_step" in df.columns:
                df[f"train/{metric}"] = df[f"train/{metric}"].add(df[f"train/{metric}_step"], fill_value=0)
            if f"train/{metric}_epoch" in df.columns:
                df[f"train/{metric}"] = df[f"train/{metric}"].add(df[f"train/{metric}_epoch"], fill_value=0)

            for m in ("val/loss", f"val/{metric}", "test/loss", f"test/{metric}"):
                if m in df.columns:
                    cols.append(m)

            return df[cols]

        ckpt_paths = (p for p in Path(logs_dir, "model_ckpts").iterdir() if "last" not in p.stem)
        return (
            pd.read_csv(logs_dir/"metrics.csv")
            .pipe(filter_columns, monitor_metric)
            .set_index(["epoch", "step"])
            .join(
                other = (pd.DataFrame({"ckpt_path": ckpt_paths})
                        .assign(epoch = lambda df: df["ckpt_path"].apply(lambda x: int(x.stem.split('_')[0].removeprefix("epoch="))))
                        .assign(step = lambda df: df["ckpt_path"].apply(lambda x: int(x.stem.split('_')[1].removeprefix("step="))))
                        .set_index(["epoch", "step"])
                        ),
                how = "outer")
            .reset_index(drop=False))

    @staticmethod
    def get_top_k_df(
        logs_dir: Path, 
        epoch: int,
        step: int,
        split: Literal["val", "test"],
        filter_by: Literal["best", "worst"],
        k: Optional[int] = None
        ) -> pd.DataFrame:
        assert filter_by in ("best", "worst"), f"{filter_by} is invalid, choose from best or worst"
        assert split in ("val", "test"), f"{split} is invalid, choose from val or test"

        dataset_csv_path = logs_dir / "dataset.csv"
        samples_csv_path = logs_dir / "eval" / f"epoch={epoch}_step={step}_{split}_samples.csv"
        print(f"Loading Dataset From: {dataset_csv_path}")
        print(f"Loading Samples From: {samples_csv_path}")

        df = (
            pd.read_csv(dataset_csv_path)
            .join(other = pd.read_csv(samples_csv_path, index_col = 0), how = "inner")
            .assign(split = split)
        )

        if filter_by == "best":
            print("best split")
            df = df.sort_values("iou", ascending=False)
            if k is not None:
                print(f"Returning the best-{k} samples, by IoU")
                df = df.iloc[:min(k, len(df))] 
            else:
                print(f"Returning the best samples, by IoU")
            df = df.reset_index(drop = True)
            return df

        else:
            print("worst split")
            df = df.sort_values("iou", ascending=True)
            if k is not None:
                print(f"Returning the worst-{k} samples, by IoU")
                df = df.iloc[:min(k, len(df))] 
            else:
                print(f"Returning the worst samples, by IoU")
            df = df.reset_index(drop = True)
            return df

    @classmethod
    def plot_epoch_eval_report(cls, confusion_matrix: NDArray, class_names: Optional[tuple[str,...]], step: int = 0, epoch: int = 0) -> Figure:
        assert confusion_matrix.ndim == 2, f"invalid shape, expected confusion_matrix.ndim = 2, received = {confusion_matrix.ndim}"
        assert confusion_matrix.shape[0] == confusion_matrix.shape[1], f"invalid shape, confusion_matrix is not a square matrix"

        fig, (left, right) =  plt.subplots(1, 2, figsize = (12, 5), width_ratios=(.7, .3))
        fig.suptitle(f"Evaluation Report, step={step}-epoch={epoch}")
        cls.plot_confusion_matrix(left, confusion_matrix);
        cls.plot_metric_table(right, cls.get_metrics_df(confusion_matrix, class_names), (1, 2));
        plt.tight_layout()
        plt.close("all")
        return fig

    @classmethod
    def plot_experiment_eval_report(cls, logs_dir: Path, monitor_metric:str, save: bool) -> None:
        "plots "
        def get_x_y(df: pd.DataFrame, x_col: str, y_col: str) -> tuple:
            view = df[[x_col, y_col]].dropna()
            x = view.iloc[:, 0].values
            y = view.iloc[:, 1].values
            return x, y 

        logs_df = cls.get_logs_df(logs_dir, monitor_metric)
        line_plots = {
            "train/loss_step": {"color": "skyblue", "linewidth": 1},
            "train/loss_epoch": {"color": "dodgerblue", "linewidth": 2},
            "val/loss": {"color": "darkorange", "linewidth": 2},
            "test/loss": {"color": "firebrick", "linewidth": 2},

            f"train/{monitor_metric}": {"color": "dodgerblue", "linewidth": 1, "linestyle": "dashed"},
            f"val/{monitor_metric}": {"color": "darkorange", "linewidth": 1, "linestyle": "dashed"},
            f"test/{monitor_metric}": {"color": "firebrick", "linewidth": 1, "linestyle": "dashed"},
        }

        scatter_plots = {
            f"test/loss": {"color": "firebrick", "marker": "."},
            f"test/{monitor_metric}": {"color": "firebrick", "marker": "."},
        }

        fig, ax = plt.subplots(1, 1, figsize = (12, 5))
        ax.grid(visible = False, axis = "x")

        for metric, params in line_plots.items():
            if metric in logs_df.columns:
                ax.plot(*get_x_y(logs_df, "step", metric), label = metric, **params)
        
        for metric, params in scatter_plots.items():
            if metric in logs_df.columns:
                ax.scatter(*get_x_y(logs_df, "step", metric), label = metric, **params)

        _, y_end = ax.get_ylim()
        ax.set_yticks(np.arange(0, max(1.05, y_end), 0.05))

        epoch_ticks = logs_df.groupby("epoch")["step"].max().tolist()
        epoch_ticks = sorted(set(epoch_ticks))
        ax.set_xticks(epoch_ticks, labels = [str(x) for x in range(len(epoch_ticks))])
        ax.xaxis.set_ticks_position("top")

        ckpt_ticks = logs_df[["step", "ckpt_path"]].dropna().iloc[:, 0].tolist()
        ckpt_axis = ax.secondary_xaxis(location=0)
        ckpt_axis.set_xticks(ckpt_ticks)
        for tick in ckpt_ticks:
            ax.axvline(tick, color = "gray", linewidth = 1, linestyle = "dashed")

        ax.legend(fontsize = 8, bbox_to_anchor=(1, 1.01))
        fig.suptitle(f"loss/{monitor_metric} vs step/epoch")
        if save:
            fig.savefig(logs_dir/"loss_vs_epoch.png")
            print(f"saved to: {logs_dir/'loss_vs_epoch.png'}")

    @staticmethod
    def plot_metric_table(ax: Axes, metrics_df: pd.DataFrame, table_scaling: tuple[float, float] = (1., 1.)):
        class_names = metrics_df.index[:-3]
        table = ax.table(
            cellText = metrics_df.round(3).values, # type: ignore
            rowLabels = tuple(f"{i}: {c}" for i, c in enumerate(class_names)) + ("accuracy", "macro avg", "weighted avg"),
            colLabels = ["precision", "recall", "jaccard", "f1", "support"],
            cellLoc = "center",
            rowLoc = "center",
            loc = "center",
            edges = "horizontal"
        )
        table.scale(*table_scaling)
        table.auto_set_font_size(False)
        table.set_fontsize(10) 
        ax.set_axis_off()

    @staticmethod
    def plot_confusion_matrix(ax: Axes, confusion_matrix: NDArray):
        _num_classes = confusion_matrix.shape[0]
        _font_size = 10 if _num_classes < 30 else 8 

        ax.grid(visible=False)
        ax.imshow(confusion_matrix, cmap = "Blues")
        ax.set_xlabel("Predicted Class", fontsize = _font_size)
        ax.set_xticks(list(range(_num_classes)))
        ax.xaxis.set_label_position("top")

        ax.set_ylabel("True Class", fontsize = _font_size)
        ax.set_yticks(list(range(_num_classes)))
        ax.yaxis.set_label_position("left")

        for r in range(_num_classes):
            for c in range(_num_classes):
                ax.text(y = r, x = c, s = str(confusion_matrix[r, c]), 
                    ha = "center", va = "center", fontsize=_font_size)

class EvaluationMetricsLogger(Callback):
    """Actually (technically) a callback whose purpose is to log additional evaluation/metrics"""

    def __init__(self) -> None:
        super().__init__()

    def __on_eval_start(self, trainer: Trainer, dataset_df: pd.DataFrame) -> None:
        self.csv_logger, self.wandb_logger = None, None
        for logger in trainer.loggers:
            if isinstance(logger, CSVLogger):
                self.csv_logger = logger
                dataset_df.to_csv(Path(self.csv_logger.log_dir, "dataset.csv"), index = False)

            elif isinstance(logger, WandbLogger):
                self.wandb_logger = logger
                self.wandb_logger.log_table(key = "dataset", dataframe = dataset_df, step = 0)

    def __on_eval_epoch_start(self) -> None:
        self.samples = {"idx": list(), "iou": list(), "dice": list()}
    
    def __on_eval_batch_end(self, outputs: torch.Tensor, inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor], num_classes: int) -> None:
        assert isinstance(inputs, tuple | list), f"expected type(input to eval_step) = list or tuple, got {type(inputs)}"
        assert isinstance(outputs, torch.Tensor), f"expected type(output of eval_step) = torch.Tensor, got {type(outputs)}"
        if len(inputs) > 2:
            preds, masks, idxs = outputs.argmax(1).detach().cpu(), inputs[1].argmax(1).detach().cpu(), inputs[2].detach().cpu()
            metric_kwargs = {"task" : "multiclass" if num_classes > 2 else "binary", "num_classes" : num_classes, "average": "macro"}
            for idx, pred, mask in zip(idxs, preds, masks):
                self.samples["idx"].append(idx.item())
                self.samples["iou"].append(jaccard_index(pred, mask, **metric_kwargs).item())
                self.samples["dice"].append(f1_score(pred, mask, **metric_kwargs).item())

    def __on_eval_epoch_end(self, trainer: Trainer, pl_module: LightningModule, confusion_matrix: ConfusionMatrix, prefix: Literal["val", "test"]) -> None:
        step, epoch = trainer.global_step, trainer.current_epoch

        samples = pd.DataFrame(self.samples) 
        confm: NDArray = confusion_matrix.compute().cpu().numpy() # type: ignore
        fig: Figure = EvaluationReport.plot_epoch_eval_report(confm, pl_module.class_names, step, epoch)
        _prefix : str = f"{prefix}/"

        pl_module.log_dict(EvaluationReport.get_metrics_dict(confm, _prefix, pl_module.class_names))
        if self.csv_logger is not None:
            _eval_dir: Path = validate_dir(self.csv_logger.log_dir, "eval")
            samples.to_csv(_eval_dir / f"epoch={epoch}_step={step}_{prefix}_samples.csv", index = False)
            fig.savefig(_eval_dir / f"epoch={epoch}_step={step}_{prefix}_eval_report.png")
            np.save(_eval_dir / f"epoch={epoch}_step={step}_{prefix}_confusion_matrix.npy", confm)
        if self.wandb_logger is not None:
            self.wandb_logger.log_table(key = f"{_prefix}samples", dataframe = samples, step = step)
            self.wandb_logger.experiment.log({f"{_prefix}eval_report": fig, "trainer/global_step": step})
            self.wandb_logger.experiment.log({f"{_prefix}confusion_matrix": confm, "trainer/global_step": step})

    def on_validation_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.__on_eval_start(trainer, trainer.datamodule.val_dataset.df) # type: ignore
        
    def on_validation_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.__on_eval_epoch_start() 
    
    def on_validation_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: torch.Tensor, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        self.__on_eval_batch_end(outputs, batch, pl_module.num_classes)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.__on_eval_epoch_end(trainer, pl_module, pl_module.val_confusion_matrix, "val")

    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.__on_eval_start(trainer, trainer.datamodule.test_dataset.df) # type: ignore
        
    def on_test_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.__on_eval_epoch_start() 
    
    def on_test_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: torch.Tensor, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        self.__on_eval_batch_end(outputs, batch, pl_module.num_classes)

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.__on_eval_epoch_end(trainer, pl_module, pl_module.test_confusion_matrix, "test")