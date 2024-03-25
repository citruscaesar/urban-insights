from pathlib import Path
import torch
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from captum.attr import LayerGradCam 
from tqdm.auto import tqdm

from etl.etl import reset_dir
# plt.rcParams["text.usetex"] = True

from typing import Optional, Any, Literal, Callable
from numpy.typing import NDArray
from torch.utils.data import Dataset, DataLoader
from matplotlib.axes import Axes
from matplotlib.figure import Figure

class Attribution:
    def __init__(
            self,
            logs_dir: Path,
            epoch: int,
            step: int,
            model: torch.nn.Module,
            dataset: Dataset,
    ):

        ckpt_path = logs_dir / "model_ckpts" / f"epoch={epoch}_step={step}.ckpt"
        state_dict = torch.load(ckpt_path)["state_dict"]
        state_dict = {k.removeprefix("model."): state_dict[k] for k in state_dict.keys()}
        self.model = model
        self.model.load_state_dict(state_dict)
        self.df = dataset.df
        self.dataloader = self.get_attr_dataloader(dataset)
        self.attribution_dir = reset_dir(logs_dir, "attribution", f"epoch={epoch}_step={step}")
        print(f"writing attributions @ [{self.attribution_dir}]")

    def get_attr_dataloader(self, dataset: Dataset) -> DataLoader:
        return DataLoader(
            dataset = dataset,
            batch_size = 1,
            shuffle = False,
        )
    
    def compute(self):
        self.model = self.model.cuda()
        attr = LayerGradCam(self.wrapper, self.model.segmentation_head[0])

        for image, mask, df_idx in tqdm(self.dataloader, desc = "Saving Attribution Maps"):
            image = image.cuda()
            image.requires_grad = True
            grads = attr.attribute(image, 1).detach().cpu()
            preds = self.model(image).detach().cpu()
            metadata = self.df.iloc[df_idx]

            self.plot_segmentation_sample(
                image = image.detach().cpu().squeeze().permute(1,2,0),
                true_mask = mask.squeeze().argmax(0),
                pred_mask = preds.squeeze().argmax(0),
                grads = grads.squeeze(),
                iou = metadata["iou"].item(),
                dice = metadata["dice"].item(),
                name = metadata["tile_name"].item()
            ).savefig(self.attribution_dir / f"{metadata['tile_name'].item().removesuffix('.tif')}.png")
        # print(image.shape, image.dtype, image.device, image.min(), image.max())
        # print(mask.shape, mask.dtype, mask.device, mask.min(), mask.max())
        # print(preds.shape, preds.dtype, preds.device, preds.min(), preds.max())
        # print(grads.shape, grads.dtype, grads.device, grads.min(), grads.max())

    def wrapper(self, img):
        pred = self.model(img)
        selected_idxs = torch.zeros_like(pred).scatter_(1, pred.argmax(1, True), 1)
        out = (selected_idxs * pred).sum(dim = (2, 3))
        return out 

    def plot_segmentation_sample(self, image: NDArray, true_mask: NDArray, pred_mask: NDArray, grads: NDArray, iou: float, dice: float, name: str) -> Figure:
        fig, ((image_ax, attribution_ax), (true_mask_ax, pred_mask_ax)) = plt.subplots(2, 2, figsize = (10, 10))
        image_ax.imshow(image)
        image_ax.grid(visible = False, axis = "both")
        image_ax.axis("off")
        image_ax.set_title("Image", fontsize = 8)
        true_mask_ax.imshow(image)
        true_mask_ax.imshow(true_mask, alpha = 0.7, cmap = "Reds")
        true_mask_ax.axis("off")
        true_mask_ax.set_title("True Mask", fontsize = 8)
        pred_mask_ax.imshow(image)
        pred_mask_ax.imshow(pred_mask, alpha = 0.7, cmap = "Reds")
        pred_mask_ax.axis("off")
        pred_mask_ax.set_title("Predicted mask", fontsize = 8)
        attribution_ax.imshow(image)
        attribution_ax.imshow(grads, alpha = .7, cmap = "Reds")
        attribution_ax.axis("off")
        attribution_ax.set_title("Attribution", fontsize = 8)
        fig.suptitle(f"{name} :: IoU={iou:.2f} Dice={dice:.2f}", fontsize = 8) # type: ignore
        plt.tight_layout()
        plt.close()
        return fig


