from typing import Any
from numpy.typing import NDArray
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio as rio

from affine import Affine
from shapely import Polygon
from skimage.measure import find_contours
from lightning import LightningModule, Trainer
from lightning.pytorch import Callback

from etl.etl import validate_dir

class InferenceCallback(Callback):
    def __init__(self, logs_dir: Path, ckpt_epoch: int, ckpt_step: int, simplify_tolerance: float = 0.5):
        """
        simplify_tolerance: tolerance passed to Douglas-Peucker
        """
        self.logs_dir = logs_dir
        self.ckpt_epoch = ckpt_epoch
        self.ckpt_step = ckpt_step
        self.simplify_tolerance = simplify_tolerance

    def on_predict_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.inference_dir = validate_dir(
            self.logs_dir, "inference", f"epoch={self.ckpt_epoch}_step={self.ckpt_step}", trainer.predict_dataloaders.dataset.split
        )
        print(f"writing outputs to: [{self.inference_dir}]")
        self.df = trainer.predict_dataloaders.dataset.df

    def on_predict_batch_start(self, trainer: Trainer, pl_module: LightningModule, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        pass

    def on_predict_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        for idx, pred in enumerate(outputs):
            metadata = self.df.iloc[batch[2][idx].cpu().item()]
            pred = (pred.detach().argmax(0, keepdim = True) * 255).cpu().numpy().astype(np.uint8)
            self._write_raster(pred, metadata)
            self._write_vector(pred, metadata)
    
    def _write_raster(self, pred: NDArray, metadata: pd.Series):
        out = rio.open(
            fp = self.inference_dir / metadata["tile_name"], mode = 'w', 
            driver = "Gtiff", dtype = "uint8", count = 1, nodata = None,
            width = metadata["wend"] - metadata["wbeg"],
            height = metadata["hend"] - metadata["hbeg"],
            crs = metadata["crs"],
            transform = Affine(metadata["xres"], 0, metadata["xoff"], 0, metadata["yres"], metadata["yoff"])
        ) 
        with out:
            out.write(pred)
    
    def _write_vector(self, pred: NDArray, metadata: pd.Series):
        transform = np.array([
            [metadata["xres"], 0, metadata["xoff"]],
            [0, metadata["yres"], metadata["yoff"]],
            [0, 0, 1]
        ], dtype = np.float32)

        polygons = list()
        for vertices in find_contours(pred.squeeze()):
            if len(vertices) < 4:
                continue
            vertices = np.matrix(vertices) # vertices = [[y1, x1], [y2, x2], ..., [yn, xn]], shape = (#vertices, 2)
            vertices[:, [0, 1]] = vertices[:, [1, 0]] # vertices = [[x1, y1], [x2, y2], ..., [xn, yn]], shape = (#vertices, 2)
            vertices = np.c_[vertices, np.ones(vertices.shape[0])] # vertices = [[x1, y1, 1], [x2, y2, 1], ..., [xn, yn, 1]], shape = (#vertices, 3)
            vertices = np.transpose(vertices) # shape = (3, #vertices)
            vertices = np.matmul(transform, vertices) # shape = (3, #vertices)
            vertices = np.transpose(vertices[:2]) # shape = (#vertices, 2)
            polygons.append(Polygon(vertices))

        (
            gpd.GeoDataFrame(geometry = polygons, crs = metadata["crs"])
            # TODO: read docs on GeoDataFrame.simplify
            # .simplify(self.simplify_tolerance)
            .to_file(
                filename = self.inference_dir / f"{metadata['tile_name'].removesuffix('.tif')}.geojson", 
                driver = "GeoJSON"
            )
        )
