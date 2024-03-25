import shutil
from pathlib import Path
import numpy as np 
import pandas as pd
import imageio.v3 as iio
import matplotlib.pyplot as plt

import zipfile
from tqdm.notebook import tqdm

from numpy.typing import NDArray

DATASET_ZIPFILES = ("berlin", "paris", "zurich", "tokyo", "chicago", "inria", "vaihingen", "potsdam")
CITIES_LATLONG = {
        "austin": (30.265842917057622, -97.74755465239008),
        "chicago": (41.874177664952654, -87.63999821844027),
        "vaihingen": (48.73014367567042, 9.104438440261527),
        "tokyo": (35.864256223966, 139.7227349836842),
        "kitsap": (47.69255762120212, -122.67612807263788),
        "tyrol-w": (47.26856603333054, 11.405558814052391),
        "vienna": (48.21066303665396, 16.378259010071375),
        "potsdam": (52.38999159662546, 13.062138562210476),
        "berlin": (52.52005606083578, 13.413606124150403),
        "zurich": (47.37592867075607, 8.544278306670433),
        "paris": (48.85512324534345, 2.3483982735980855),
    }

def plot_image_mask(left: NDArray, right: NDArray) -> None:
    _, (l, r) = plt.subplots(1,2, figsize = (10, 20))
    l.imshow(left)
    r.imshow(right, cmap = "gray")

def get_tiled_view(image: NDArray, kernel: tuple[int, int, int]):
    """
    Return a non-overlapping sliding window view of image 

    image.shape: (height, width, num_channels)
    kernel.shape: (kernel_height, kernel_width, num_kernel_channels) 

    return.shape (#Tiles, K_H, K_W, K_C)
    """
    return (
        sliding_window_view(image, kernel[:2], (0,1)) #type: ignore
        [::kernel[0], ::kernel[1]]
        .reshape(-1, kernel[2], kernel[0], kernel[1])
        .transpose(0, 2, 3, 1)
        .squeeze()
    )

def tile_dataset(tile: tuple[int,int], image_dir: Path, mask_dir: Path, tiled_image_dir: Path, tiled_mask_dir: Path):
    filenames = sorted([f.name for f in image_dir.iterdir()])
    for filename in tqdm(filenames):
        images = get_tiled_view(
            image = iio.imread(image_dir/filename).squeeze(),
            kernel = (tile[0], tile[1], 3)
        )
        masks = get_tiled_view(
            image = iio.imread(mask_dir/filename).squeeze(),
            kernel = (tile[0], tile[1], 1)
        )
        name = filename.split(".")[0]
        extn = filename.split(".")[-1]
        try:
            for idx, (image, mask) in enumerate(zip(images, masks)):
                print(tiled_image_dir/f"{name}_{idx}.{extn}") #type: ignore
                print(tiled_mask_dir/f"{name}_{idx}.{extn}") #type: ignore
        except:
            print(filename)
            continue

def extract_image(src_path, dest_path, zipfile_object) -> None:
    with open(dest_path, "wb") as dst:
        with zipfile_object.open(src_path, "r") as src:
            shutil.copyfileobj(src, dst)

def transformation_strategy_cityosm(dataset_zip_path: Path, image_dir: Path, mask_dir: Path) -> None:
    with zipfile.ZipFile(dataset_zip_path) as zf:
        filenames = [x.removesuffix("_image.png") for x in zf.namelist()[1:] if "_image" in x]
        for filename in tqdm(sorted(filenames), desc = f"{dataset_zip_path.stem.capitalize()} Progress"):
            image_src_path = f"{filename}_image.png"
            image_dst_path = image_dir/f"{filename.split('/')[-1]}.png"

            mask_src_path = zipfile.Path(dataset_zip_path) / f"{filename}_labels.png"
            mask_dst_path = mask_dir/f"{filename.split('/')[-1]}.png"

            extract_image(image_src_path, image_dst_path, zf) 

            # Mask[:, :, 0] = Road
            # Mask[:, :, 1] = Building and Road
            # Mask[:, :, 2] = Building 
            mask = iio.imread(str(mask_src_path), extension=".png")
            mask = mask[:, :, 2]
            mask = np.where(mask==255, 0, 255).astype(np.uint8)
            iio.imwrite(mask_dst_path, mask, extension=".png")

def transformation_strategy_vaihingen(dataset_zip_path: Path, image_dir: Path, mask_dir: Path) -> None:
    with zipfile.ZipFile(dataset_zip_path) as outer:
        
        images_zip_bytes = outer.open("Vaihingen/ISPRS_semantic_labeling_Vaihingen.zip")
        with zipfile.ZipFile(images_zip_bytes) as inner:
            filenames = sorted([x for x in inner.namelist() if "top/top_mosaic_09cm" in x and not x.endswith('/')])
            for image_src_path in tqdm(filenames, desc = "Vaihingen Image Files Progress"): 
                image_dst_path = image_dir/f"vaihingen{image_src_path.removeprefix('top/top_mosaic_09cm_area')}"
                extract_image(image_src_path, image_dst_path, inner)

        masks_zip_bytes = outer.open("Vaihingen/ISPRS_semantic_labeling_Vaihingen_ground_truth_COMPLETE.zip")
        with zipfile.ZipFile(masks_zip_bytes) as inner:
            for filename in tqdm(sorted(inner.namelist()), desc = "Vaihingen Mask Files Progress"):
                mask_dst_path = mask_dir/f"vaihingen{filename.removeprefix('top_mosaic_09cm_area')}"
                # mask[:, :, 0] = vegetation and building 
                # mask[:, :, 1] = building
                # mask[:, :, 2] = vegetation 
                mask = iio.imread(inner.open(filename, 'r')).squeeze() #type: ignore
                mask = mask[:, :, 1]
                mask = np.where(mask==255, 0, 255).astype(np.uint8)
                iio.imwrite(mask_dst_path, mask, extension=".tif")

def transformation_strategy_potsdam(dataset_zip_path: Path, image_dir: Path, mask_dir: Path) -> None:
    with zipfile.ZipFile(dataset_zip_path) as outer:

        images_zip_bytes = outer.open("Potsdam/2_Ortho_RGB.zip")
        with zipfile.ZipFile(images_zip_bytes) as inner:
            filenames = sorted([x for x in inner.namelist() if x.endswith(".tif")])
            for image_src_path in tqdm(filenames, desc = "Potsdam Image Files Progress"):
                image_dest_filename = image_src_path.removeprefix("2_Ortho_RGB/top_potsdam_").removesuffix("_RGB.tif")
                image_dest_filename = image_dir/f"potsdam{''.join(image_dest_filename.split('_'))}.tif"
                extract_image(image_src_path, image_dest_filename, inner)
        
        masks_zip_path = outer.open("Potsdam/5_Labels_all.zip")
        with zipfile.ZipFile(masks_zip_path) as inner:
            filenames = sorted([x for x in inner.namelist() if x.endswith(".tif")])
            for mask_src_path in tqdm(filenames, desc = "Potsdam Mask Files Progress"): 
                mask_dest_filename = mask_src_path.removeprefix("top_potsdam_").removesuffix("label.tif")
                mask_dest_filename = mask_dir/f"potsdam{''.join(mask_dest_filename.split('_'))}.tif"

                # mask[:, :, 0] = background 
                # mask[:, :, 1] = building
                # mask[:, :, 2] = no idea
                mask = iio.imread(inner.open(mask_src_path)).squeeze() # type: ignore
                mask = mask[:, :, 1]
                mask = np.where(mask==255, 0, 255).astype(np.uint8)
                iio.imwrite(mask_dest_filename, mask, extension=".tif")

