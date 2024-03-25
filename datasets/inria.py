import os
import shutil
import h5py
import zipfile
import torch
import pandas as pd
import numpy as np
import imageio.v3 as iio
import rasterio as rio
from rasterio.windows import Window
from pathlib import Path

from litdata import optimize, StreamingDataset
from tqdm.auto import tqdm
from etl.etl import validate_dir
from etl.extract import extract_multivolume_archive
from torchvision.transforms.v2 import Transform, Compose, ToImage, ToDtype, Identity
from torchvision.datasets.utils import download_url
from typing import Optional, Literal

class InriaSegmentation(torch.utils.data.Dataset):
    SUPERVISED_LOCATIONS = ("austin", "chicago", "kitsap", "vienna", "tyrol-w")
    UNSUPERVISED_LOCATIONS = ("bellingham", "bloomington", "innsbruck", "sfo", "tyrol-e")
    URLS = ("https://files.inria.fr/aerialimagelabeling/aerialimagelabeling.7z.001",
            "https://files.inria.fr/aerialimagelabeling/aerialimagelabeling.7z.002",
            "https://files.inria.fr/aerialimagelabeling/aerialimagelabeling.7z.003",
            "https://files.inria.fr/aerialimagelabeling/aerialimagelabeling.7z.004",
            "https://files.inria.fr/aerialimagelabeling/aerialimagelabeling.7z.005")
    DATASET_ARCHIVE_NAME = "NEW2-AerialImageDataset.zip"
    NUM_CLASSES = 2
    CLASS_NAMES = ("Background", "Foreground")
    NAME = "urban_footprint"
    TASK = "segmentation"
    DEFAULT_IMAGE_TRANSFORM = Compose([ToImage(), ToDtype(torch.float32, scale=True)])
    DEFAULT_TARGET_TRANSFORM = Compose([ToImage(), ToDtype(torch.float32, scale=False)])
    DEFAULT_COMMON_TRANSFORM = Identity()
    EYE = np.eye(NUM_CLASSES, dtype = np.uint8)
    SCENE_SHAPE = (5000, 5000, 3)

    @classmethod
    def download(cls, root: Path, low_storage: bool = False):
        r"""download dataset from saved URLs and write 'NEW2-AerialImageDataset.zip' to provided root directory
        Parameters
        ----------
        root: Path
            directory to store dataset in; multivolume .7zip files are stored in a (temporary) subdirectory "downloads"
        low_storage: bool, default False
            if downloaded multivolume .7zip files should be deleted after dataset archive has been saved to root
        """
        downloads = validate_dir(root, "downloads")

        # TODO: Use Async? Will it be any faster?
        print(f"Downloading .7z archives to {downloads}")
        for url in tqdm(cls.URLS):
            download_url(url, str(downloads))

        dataset_archive_path = (root / cls.DATASET_ARCHIVE_NAME)
        print(f"Extracting dataset archive to {dataset_archive_path}")
        extract_multivolume_archive(downloads / "aerialimagelabeling.7z", root)
        if low_storage:
            print(f"Deleting downloaded .7z archives from {downloads}")
            shutil.rmtree(str(downloads))
    
    @classmethod
    def get_metadata_df(cls, root: Path, save: bool = False, split: Literal["sup", "unsup"] = "sup") -> pd.DataFrame:
        r""" generate metadata dataframe by reading dimensions, crs and transformation matrices from 'NEW2-AerialImageDataset.zip'
        Parameters
        ----------
        root: Path
            directory where dataset archive is located
        split: str, default = "sup"
            supervised or unsupervised dataset
        save: bool, default = False 
            whether to save metadata.csv to dataset root or not
        """
        assert split in ("sup", "unsup"), f"{split} is invalid, choose from (sup, unsup)"

        def get_crs(df: pd.DataFrame, root: Path, split: Literal["sup", "unsup"]):
            crs_df = pd.DataFrame(columns = ["crs", "xoff", "yoff", "xres", "yres"])  
            # TODO: check how this str(path) works on windows?
            if split == "sup":
                zipfile_path = f"zip+file://{str(root/cls.DATASET_ARCHIVE_NAME)}!/AerialImageDataset/train/images/"
            else:
                zipfile_path = f"zip+file://{str(root/cls.DATASET_ARCHIVE_NAME)}!/AerialImageDataset/test/images/"
            for _, row in tqdm(df.iterrows(), total = len(df), desc = "Collecting CRS and Projections"):
                with rio.open(zipfile_path + row["scene_name"]) as raster:
                    # NOTE: losing decimals here? might affect location accuracy
                    xres, _, xoff, _, yres, yoff, _, _, _ = tuple(raster.transform)
                    crs_df.loc[len(crs_df.index)] = [str(raster.crs), xoff, yoff, xres, yres]
            return pd.concat([df, crs_df], axis = 1)

        get_scene_names = lambda locs: [f"{loc}{n}.tif" for loc in locs for n in range(1, 36)]
        df = (
            pd.DataFrame({"scene_name": get_scene_names(cls.SUPERVISED_LOCATIONS if split == "sup" else cls.UNSUPERVISED_LOCATIONS)})
            .pipe(get_crs, root, split)
            .assign(dataset_name = "inria")
            .assign(scene_idx = lambda df: df.index)
            [["scene_idx", "dataset_name", "scene_name", "crs", "xoff", "yoff", "xres", "yres"]]
        )
        if save:
            df.to_csv(root / "metadata.csv", index = False)
        return df

    @classmethod
    def get_dataset_df(
            cls,
            metadata_df: pd.DataFrame,
            random_seed: int = 42,
            test_split: float = 0.2,
            val_split: float = 0.1,
            tile_size: Optional[tuple[int, int]] = None, 
            tile_stride: Optional[tuple[int, int]] = None,
            **kwargs
        ) -> pd.DataFrame:
        r""" generate dataset dataframe with train-val-test splits and optional tiling, used by pytorch dataset to load samples 
        Parameters
        ----------
        metadata_df: DataFrame
            dataframe with scene_idx, dataset_name, scene_name, width, height, crs, xoff, yoff, xres, yres
            generated using cls.get_metadata_df 
        random_seed: int 
            used to randomly sample dataset to generate train-val-test splits
        test_split: float, between [0, 1]
            proportion of dataset to use as test data
        val_split: float, between [0, 1]
            proportion of dataset to use as validation data
        tile_size: tuple[int, int], optional
            size (x, y) of the sliding window (kernel) used to draw samples 
        tile_stride: tuple[int, int], optional
            stride (x, y) of the sliding window (kernel) used to draw samples 
        if tile_size and tile_stride are not provided, dataset is not tiled

        **kwargs: dict[str, Any]
            to handle additional params and laziness
        """
        def get_loc(scene_name: str) -> str:
            for idx, char in enumerate(scene_name):
                if char.isnumeric():
                    return scene_name[:idx]
            return scene_name

        def set_splits(df: pd.DataFrame, random_seed: int, test_split: float, val_split: float) -> pd.DataFrame:
            """
            Sample train-val-test data in a stratified (proportionate) mannner, based on column 'loc'.
            This will pick eval_split proportion of samples from each class, not eval_split proportion
            from the entire dataset. Basically #eval samples from the ith class = eval_split * #samples in ith class.
            """

            test = (df
                    .groupby("loc", group_keys=False)
                    .apply(lambda x: x.sample(frac = test_split, random_state = random_seed, axis = 0), include_groups = False)
                    .assign(split = "test"))

            val = (df
                    .drop(test.index, axis = 0)
                    .groupby("loc", group_keys=False)
                    .apply(lambda x: x.sample(frac = val_split / (1-test_split), random_state = random_seed, axis = 0), include_groups = False)
                    .assign(split = "val"))

            train = (df
                    .drop(test.index, axis = 0)
                    .drop(val.index, axis = 0)
                    .assign(split = "train"))

            return (pd.concat([train, val, test])
                        .sort_index()
                        .drop("loc", axis = 1))
        
        def get_num_windows(length: int, kernel: int, stride: int) -> int:
            return (length - kernel - 1) // stride + 2

        if tile_size is not None and tile_stride is None:
            raise ValueError("tile_stride not provided")

        elif tile_size is None and tile_stride is not None:
            raise ValueError("tile_size is not provided")

        elif tile_size is None and tile_stride is None:
            return (
                metadata_df
                .assign(loc = lambda df: df["scene_name"].apply(lambda x: get_loc(x)))
                .pipe(set_splits, random_seed, test_split, val_split)
                .assign(hbeg = 0)
                .assign(hend = cls.SCENE_SHAPE[0])
                .assign(wbeg = 0)
                .assign(wend = cls.SCENE_SHAPE[1])
                .assign(tile_name = lambda df: df.apply(
                    lambda x: f"{x['scene_name'].removesuffix('.tif')}_{x['wbeg']}_{x['wend']}_{x['hbeg']}_{x['hend']}.tif",
                    axis = 1))
                [["dataset_name", "scene_idx", "scene_name", "tile_name", "split", "wbeg", "wend", "hbeg", "hend", "crs", "xoff", "yoff", "xres", "yres"]]
            )

        else:
            metadata_df = (
                metadata_df
                .assign(loc = lambda df: df["scene_name"].apply(lambda x: get_loc(x)))
                .pipe(set_splits, random_seed, test_split, val_split))

            tile_data = {k: list() for k in ["scene_idx", "tile_name", "hbeg", "hend", "wbeg", "wend"]} 
            for _, row in metadata_df.iterrows():
                scene_name = Path(row["scene_name"])
                for x in range(0, get_num_windows(cls.SCENE_SHAPE[0], tile_size[0], tile_stride[0])):
                    for y in range(0, get_num_windows(cls.SCENE_SHAPE[1], tile_size[1], tile_stride[1])):
                        hbeg, wbeg = x * tile_stride[0], y * tile_stride[1]
                        hend, wend = hbeg + tile_size[0], wbeg + tile_size[1]
                        tile_data["scene_idx"].append(row["scene_idx"])
                        tile_data["tile_name"].append(f"{scene_name.stem}_{wbeg}_{wend}_{hbeg}_{hend}{scene_name.suffix}")
                        tile_data["hbeg"].append(hbeg)
                        tile_data["hend"].append(hend)
                        tile_data["wbeg"].append(wbeg)
                        tile_data["wend"].append(wend)

            return (
                pd.DataFrame(tile_data)
                .merge(metadata_df, on = "scene_idx", how = "outer")
                .assign(xoff = lambda df: df.apply(lambda x: x["xoff"] + (x["xres"] * x["wbeg"]), axis = 1))
                .assign(yoff = lambda df: df.apply(lambda x: x["yoff"] + (x["yres"] * x["hbeg"]), axis = 1))
                [["dataset_name", "scene_idx", "scene_name", "tile_name", "split", "wbeg", "wend", "hbeg", "hend", "crs", "xoff", "yoff", "xres", "yres"]]
            )
        
    @classmethod
    def write_to_files(cls, root: Path, target: Path, df: pd.DataFrame, **kwargs):
        r"""
        Parameters
        ----------
        root: Path
            Directory where "NEW2-AerialImageDataset.zip" is located

        target: Path
            Directory to write prepared files to, additional subdirectories are
            created for supervised and unsupervised tiles 

        df: DataFrame
            DataFrame with appropriate columns containing train-val-test splits
            df.columns = {scene_name, tile_name, split, hbeg, hend, wbeg, wend}

        **kwargs: dict[str, Any], optional
            for being lazy 
        """

        def extract_tile(src: str, tgt: str, hbeg:int, hend:int, wbeg:int, wend:int):
            H, W = int(hend-hbeg), int(wend-wbeg)
            tile_window = Window(wbeg, hbeg, W, H)
            with rio.open(src) as scene:
                tile_profile = scene.profile.copy() 
                tile_profile.update(width = W, height = H, transform = scene.window_transform(tile_window))
                with rio.open(tgt, 'w', **tile_profile) as tile:
                    tile.write(scene.read(window = tile_window))

        DATASET_ZIP = root / "NEW2-AerialImageDataset.zip"
        assert DATASET_ZIP.is_file(), f"Dataset Archive Missing @ [{DATASET_ZIP}]"
        IMAGES = validate_dir(target, "sup", "images")
        MASKS = validate_dir(target, "sup", "masks")
        UNSUP = validate_dir(target, "unsup", "images")

        for _, row in tqdm(df.iterrows(), desc = "Writing to Files", total = len(df)):
            tile_dims = dict(hbeg = row["hbeg"], hend = row["hend"], wbeg = row["wbeg"], wend = row["wend"])

            if row["split"] == "unsup":
                extract_tile(
                    src = f"zip+file://{str(DATASET_ZIP)}!/AerialImageDataset/test/images/{row['scene_name']}",
                    tgt = str(UNSUP/row['tile_name']), 
                    **tile_dims
                )
            else:
                extract_tile(
                    src = f"zip+file://{str(DATASET_ZIP)}!/AerialImageDataset/train/images/{row['scene_name']}",
                    tgt = str(IMAGES/row['tile_name']),
                    **tile_dims,
                )
                extract_tile(
                    src = f"zip+file://{str(DATASET_ZIP)}!/AerialImageDataset/train/gt/{row['scene_name']}",
                    tgt = str(MASKS/row['tile_name']),
                    **tile_dims,
                )

    @classmethod
    def write_to_hdf(cls, root: Path, target: Path, **kwargs) -> None:
        r"""
        Parameters
        ----------
        root: Path
            Directory where "NEW2-AerialImageDataset.zip" is located
        target: Path
            Directory to write prepared hdf5 files to 
        **kwargs: dict[str, Any], optional
            to accomodate laziness
        """

        # bits * numimages * width * height * num_bands(RGB+Mask)
        # size_in_bits = 8 * 180 * 5000 * 5000 * (3+3+2) 
        # size_in_gigabytes = size_in_bits / (8 * 1024 * 1024 * 1024)
        # ~33.52GB
        target = validate_dir(target)
        dataset_zip = root / "NEW2-AerialImageDataset.zip"
        assert dataset_zip.is_file(), f"Dataset Archive Missing @ [{dataset_zip}]"

        with h5py.File(target / f"inria.h5", 'w') as f:
            scene_names = [f"{loc}{n}.tif" for loc in cls.SUPERVISED_LOCATIONS for n in range(1, 36)]
            dataset = f.create_dataset("supervised", (len(scene_names), cls.SCENE_SHAPE[0], cls.SCENE_SHAPE[1], 5), dtype = "uint8")
            for idx, scene_name in tqdm(enumerate(scene_names), total = len(scene_names), desc = "Writing Supervised Dataset"):
                image = iio.imread(dataset_zip / f"AerialImageDataset/train/images/{scene_name}")
                mask = iio.imread(dataset_zip / f"AerialImageDataset/train/gt/{scene_name}")
                mask = cls.EYE[np.clip(mask, 0, 1)]
                dataset[idx] = np.dstack([image, mask])

        with h5py.File(target / f"inria_unsupervised.h5", 'w') as f:
            scene_names = [f"{loc}{n}.tif" for loc in cls.UNSUPERVISED_LOCATIONS for n in range(1, 36)]
            dataset = f.create_dataset("unsupervised", (len(scene_names), cls.SCENE_SHAPE[0], cls.SCENE_SHAPE[1], 3), dtype = "uint8")
            for idx, scene_name in tqdm(enumerate(scene_names), total = len(scene_names), desc = "Writing Unsupervised Dataset"):
                dataset[idx] = iio.imread(dataset_zip / f"AerialImageDataset/test/images/{scene_name}")

    @classmethod
    def write_to_litdata(
            cls, 
            root: Path,
            target: Path,
            dataset_df: pd.DataFrame,
            num_workers: Optional[int] = None,
            shard_size_in_mb: int = 512,
            **kwargs) -> None:
        """
        Parameters
        -----
        root: Path
            Directory where "NEW2-AerialImageDataset.zip" is located
        target: Path
            Parent directory to store prepared shards, usually Path.home() / "shards" / "dataset_name", 
            shards are stored in subdirectories within this directory
        dataset_df: DataFrame, optional
            DataFrame with scene_idx, scene_name, split, hbeg, hend, wbeg, wend, generated using
            cls.get_dataframe_df()
        num_workers: int
            number of processes used to encode dataset, defaults to os.cpu_count() 
        shard_size_in_mb: int 
            size of each chunk in megabytes, defaults to 512 
        **kwargs: dict[str, Any]
            for laziness
        """

        DATASET = root / cls.DATASET_ARCHIVE_NAME / "AerialImageDataset"
        assert DATASET.parent.is_file(), f"Dataset Archive Missing @ [{DATASET.parent}]"

        def read_image(src_uri: str, hbeg:int, hend:int, wbeg:int, wend:int):
            image = iio.imread(src_uri, extension = ".tif")
            H, W = image.shape[0], image.shape[1]
            image = image[hbeg: min(hend, H), wbeg: min(wend, W)].copy()
            if hend > H or wend > W:
                if image.ndim == 2: 
                    image = np.pad(image, ((0, max(0, hend - H)), (0, max(0, wend - W))), "constant", constant_values = 0).copy()
                else:
                    image = np.pad(image, ((0, max(0, hend - H)), (0, max(0, wend - W)), (0, 0)), "constant", constant_values = 0).copy()
            return image

        def encode_supervised_samples(inpt):
            idx, row = inpt
            tile_dims = row["hbeg"], row["hend"], row["wbeg"], row["wend"]
            image = read_image(DATASET/"train"/"images"/row["scene_name"], *tile_dims)
            mask =  read_image(DATASET/"train"/"gt"/row["scene_name"], *tile_dims)
            mask = cls.EYE[np.clip(mask, 0, 1)]
            return {
                "scene_idx": idx, 
                "image": image,
                "mask": mask,
            } 

        # def encode_unsupervised_samples(inpt):
            # idx, row = inpt
            # tile_dims = row["hbeg"], row["hend"], row["wbeg"], row["wend"]
            # image = read_image(DATASET/"test"/"images"/row["scene_name"], *tile_dims)
            # return {
                # "scene_idx": idx, 
                # "image": image,
            # } 
        
        dataset_df.to_csv(target / "metadata.csv")
        for split in ("train", "val", "test"):
            optimize(
                fn = encode_supervised_samples, # if split!="unsup" else encode_unsupervised_samples,
                inputs = list(dataset_df[dataset_df["split"] == split].iterrows()),
                output_dir = str(target / split),
                num_workers = num_workers if num_workers is not None else os.cpu_count(),
                chunk_bytes = shard_size_in_mb * 1024 * 1024,
            )


    @staticmethod
    def _show_tiles_along_one_dim(dim_len: int, kernel: int, stride: int, padding: Optional[int] = None) -> None:
        num_tiles = (dim_len - kernel - 1) // stride + 2
        padding = (num_tiles - 1) * stride + kernel - dim_len if padding is None else padding

        print(f"Can Create {num_tiles} Windows")
        print(f"If Image Is Padded by: {padding} Pixels\n")

        for tile_idx in range(0, num_tiles):
            print(f"Tile #{tile_idx} -> [{tile_idx * stride}:{(tile_idx * stride + kernel)})") 

    @staticmethod
    def _subset_df(df: pd.DataFrame, split: str):
        if split == "trainval":
            return (df[(df.split == "train") | (df.split == "val")].reset_index(drop=True)) # type: ignore
        return (df[df.split == split].reset_index(drop=True))

class InriaImageFolder(InriaSegmentation):
    def __init__(
            self,
            root: Path,
            df: Optional[pd.DataFrame] = None,
            split: str = "train",
            test_split: float = 0.2,
            val_split: float = 0.2,
            random_seed: int = 42,
            tile_size: Optional[tuple[int, int]] = None,
            tile_stride: Optional[tuple[int, int]] = None,
            image_transform: Optional[Transform] = None,
            target_transform: Optional[Transform] = None,
            common_transform: Optional[Transform] = None,
            **kwargs,
        ) -> None:

        assert split in ("train", "val", "test", "trainval"), "Invalid Split"
        self.root = root
        self.split = split
        self.image_transform = image_transform or self.DEFAULT_IMAGE_TRANSFORM
        self.target_transform = target_transform or self.DEFAULT_TARGET_TRANSFORM 
        self.common_transform = common_transform or self.DEFAULT_COMMON_TRANSFORM 

        experiment_kwargs = {
            "metadata_df": pd.read_csv(root/"metadata.csv"),
            "random_seed" : random_seed,
            "val_split": val_split,
            "test_split": test_split,
        }

        if isinstance(df, pd.DataFrame):
            assert {"scene_name", "split", "hbeg", "hend", "wbeg", "wend"}.issubset(df.columns), "incorrect dataframe schema"
            print(f"{split} custom dataset @ [{self.root}]")
            self.df = df
        elif tile_size is None and tile_stride is None:
            print(f"{split} scene dataset @ [{self.root}]")
            self.df = self.get_dataset_df(**experiment_kwargs)
        else:
            print(f"{split} tiled dataset @ [{self.root}]")
            self.df = self.get_dataset_df(**experiment_kwargs, tile_size = tile_size, tile_stride = tile_stride)
        
        self.df = (
            self.df
            .assign(image_path = lambda df: df.apply(lambda x: str(Path("sup", "images", x["scene_name"])), axis = 1))
            .assign(mask_path = lambda df: df["image_path"].apply(lambda x: str(Path(str(x).replace("images", "masks")))))
            .assign(df_idx = lambda df: df.index)
        )
        self.split_df  = (
            self.df
            .pipe(self._subset_df, split)
            .pipe(self._prefix_root_to_df, root)
        )
    
    def _prefix_root_to_df(self, df: pd.DataFrame, root: Path) -> pd.DataFrame:
        return (
            df
            .assign(image_path = lambda df: df["image_path"].apply(lambda x: str(root/x)))
            .assign(mask_path = lambda df: df["mask_path"].apply(lambda x: str(root/x)))
        ) 
    
    def __len__(self):
        return len(self.split_df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:

        # TODO: Can all these extra lines be removed? Test
        # NumPy slicing creates a view instead of a copy as in the case of built-in Python sequences such as string, tuple and list.
        # Care must be taken when extracting a small portion from a large array which becomes useless after the extraction,
        # because the small portion extracted contains a reference to the large original array whose memory will not be released
        # until all arrays derived from it are garbage-collected.
        # In such cases an explicit copy() is recommended.

        row = self.split_df.iloc[idx]
        H, W, hbeg, hend, wbeg, wend = self.SCENE_SHAPE[0], self.SCENE_SHAPE[1], row["hbeg"], row["hend"], row["wbeg"], row["wend"]

        image_scene = iio.imread(row["image_path"])
        image = image_scene[hbeg: min(hend, H), wbeg: min(wend, W), :].copy()
        del image_scene

        mask_scene = iio.imread(row["mask_path"])
        mask = mask_scene[hbeg: min(hend, H), wbeg: min(wend, W)].copy()
        del mask_scene

        if hend > H or wend > W:
            image = np.pad(image, ((0, max(0, hend - H)), (0, max(0, wend - W)), (0, 0)), "constant", constant_values = 0).copy()
            mask = np.pad(mask, ((0, max(0, hend - H)), (0, max(0, wend - W))), "constant", constant_values = 0).copy()

        mask = self.EYE[np.clip(mask, 0, 1)]

        image = self.image_transform(image)
        mask = self.target_transform(mask)

        if self.split == "train":
            image, mask = self.common_transform([image, mask])
        return image, mask, row["df_idx"]

class InriaHDF5(InriaSegmentation):
    def __init__(
            self,
            root: Path,
            df: Optional[pd.DataFrame] = None,
            split: str = "train",
            test_split: float = 0.2,
            val_split: float = 0.2,
            random_seed: int = 42,
            tile_size: Optional[tuple[int, int]] = None,
            tile_stride: Optional[tuple[int, int]] = None,
            image_transform: Optional[Transform] = None,
            target_transform: Optional[Transform] = None,
            common_transform: Optional[Transform] = None,
            **kwargs,
        ) -> None:
        assert root.is_file() and (root.suffix == ".h5" or root.suffix == ".hdf5"), f"{root} does not point to an .h5/.hdf5 file"
        assert split in ("train", "val", "test", "unsup", "trainval"), f"provided split [{split}] is invalid"
        self.root = root
        self.split = split
        self.image_transform = image_transform or self.DEFAULT_IMAGE_TRANSFORM
        self.target_transform = target_transform or self.DEFAULT_TARGET_TRANSFORM 
        self.common_transform = common_transform or self.DEFAULT_COMMON_TRANSFORM 

        experiment_kwargs = {
            "metadata_df": pd.read_csv(root.parent/"metadata.csv"),
            "random_seed" : random_seed,
            "val_split": val_split,
            "test_split": test_split,
        }

        if isinstance(df, pd.DataFrame):
            assert {"scene_name", "split", "hbeg", "hend", "wbeg", "wend"}.issubset(df.columns), "incorrect dataframe schema"
            print(f"{split} custom dataset @ [{self.root}]")
            self.df = df
        elif tile_size is None and tile_stride is None:
            print(f"{split} scene dataset @ [{self.root}]")
            self.df = self.get_dataset_df(**experiment_kwargs)
        else:
            print(f"{split} tiled dataset @ [{self.root}]")
            self.df = self.get_dataset_df(**experiment_kwargs, tile_size = tile_size, tile_stride = tile_stride)

        self.df = self.df.assign(df_idx = lambda df: df.index)
        self.split_df = self.df.pipe(self._subset_df, split)    

    def __len__(self):
        return len(self.split_df)
        
    def __getitem__(self, idx):
        row = self.split_df.iloc[idx]
        H, W = self.SCENE_SHAPE[0], self.SCENE_SHAPE[1]
        hbeg, hend, wbeg, wend = row["hbeg"], row["hend"], row["wbeg"], row["wend"]
        with h5py.File(self.root, mode = "r") as f:
            image_mask = f["supervised"][row["scene_idx"], hbeg: min(hend, H), wbeg: min(wend, W)] 
        image = image_mask[:, :, :3].copy()
        mask = image_mask[:, :, 3:].copy()
        del image_mask
        if hend > H or wend > W: 
            image = np.pad(image, ((0, max(0, hend - H)), (0, max(0, wend - W)), (0, 0)), "constant", constant_values = 0).copy()
            mask = np.pad(mask, ((0, max(0, hend - H)), (0, max(0, wend - W)), (0, 0)), "constant", constant_values = 0).copy()
        image, mask = self.image_transform(image), self.target_transform(mask)
        if self.split == "train":
            image, mask = self.common_transform([image, mask])
        return image, mask, row["df_idx"]

class InriaLitData(StreamingDataset, InriaSegmentation):
    def __init__(
            self,
            root: Path,
            split: Literal["train", "val", "test"] = "train",
            random_seed: int = 42,
            image_transform: Optional[Transform] = None,
            target_transform: Optional[Transform] = None,
            common_transform: Optional[Transform] = None,
            shuffle: bool = False,
            cache_limit: int | str = "30GB",
            **kwargs,
    ) -> None:
        assert split in ("train", "val", "test"), "Invalid Split"
        root = str(root/split) if isinstance(root, Path) else f"{root.removesuffix('/')}/{split}/"
        self.root = Path(root)
        self.split = split
        self.df = pd.read_csv(self.root / "metadata.csv")
        print(f"{self.split} scene dataset @ [{self.root}]")

        self.image_transform = image_transform or self.DEFAULT_IMAGE_TRANSFORM
        self.target_transform = target_transform or self.DEFAULT_TARGET_TRANSFORM 
        self.common_transform = common_transform or self.DEFAULT_COMMON_TRANSFORM

        super().__init__(
            input_dir = root,
            shuffle = shuffle,
            seed = random_seed,
            max_cache_size = cache_limit,
        )
    
    def __len__(self):
        return super().__len__()

    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        image, mask = self.image_transform(data["image"].copy()), self.target_transform(data["mask"].copy())
        if self.split == "train":
            image, mask = self.common_transform([image, mask])
        return image, mask, data["scene_idx"]



