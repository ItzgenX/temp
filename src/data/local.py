import json
import os
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
import torch
import numpy as np



def sort_key(p: Path):
    try:
        return int(p.stem)
    except:
        return p.stem


class ImageFolderDataset(Dataset):
    def __init__(self, directory: Path, transform, caption_from_name: bool, caption_prefix: str):
        """
        Args:
            directory (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.directory = directory
        self.transform = transform
        self.image_paths = [directory / file for file in os.listdir(directory) if file.endswith(("jpg", "jpeg", "png"))]
        self.image_paths.sort(key=sort_key)
        self.caption_from_name = caption_from_name
        self.caption_prefix = caption_prefix

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image_path = self.image_paths[idx]
        txt_path = image_path.with_suffix(".txt")

        if self.caption_from_name:
            label = self.caption_prefix + image_path.stem.split("_")[0].replace("-", " ")
        else:
            try:
                with open(txt_path, "r") as f:
                    label = f.read()
            except:
                label = ""

        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return {"jpg": image, "caption": label}


class ZipDataset(Dataset):
    def __init__(self, datasets: list[ImageFolderDataset]):
        # Ensure all datasets have the same length
        assert all(len(datasets[0]) == len(d) for d in datasets), "Datasets must all be the same length!"
        self.datasets = datasets

    def __len__(self):
        return len(self.datasets[0])

    def __getitem__(self, idx: int):
        # Return a tuple containing elements from each dataset at the given index
        if len(self.datasets) == 1:
            return self.datasets[0][idx]

        return tuple(d[idx] for d in self.datasets)


class ImageDataModule:
    def __init__(
        self,
        directories: list[str],
        transform: list,
        val_directories: list[str] = [],
        batch_size: int = 32,
        val_batch_size: int = 1,
        workers: int = 4,
        val_workers: int = 1,
        caption_from_name: bool = False,
        caption_prefix: str = "",
    ):
        super().__init__()

        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.workers = workers
        self.val_workers = val_workers

        project_root = Path(os.path.abspath(__file__)).parent.parent.parent

        self.train_dataset = ZipDataset(
            [
                ImageFolderDataset(
                    directory=Path(project_root, d),
                    transform=transforms.Compose(transform),
                    caption_from_name=caption_from_name,
                    caption_prefix=caption_prefix,
                )
                for d in directories
            ]
        )

        self.val_dataset = ZipDataset(
            [
                ImageFolderDataset(
                    directory=Path(project_root, d),
                    transform=transforms.Compose(transform),
                    caption_from_name=caption_from_name,
                    caption_prefix=caption_prefix,
                )
                for d in val_directories
            ]
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.val_batch_size, shuffle=False, num_workers=self.val_workers)


# ============================================================================ #
#  DEPTH-AWARE DATASET CLASSES (for pre-computed depth map training)           #
# ============================================================================ #

class DepthImageFolderDataset(Dataset):
    """
    Dataset that loads an image AND its pre-computed depth map together.

    MOTIVATION:
        The original ImageFolderDataset only loads the RGB image.  During training
        the depth estimator (DPT) then runs on every image every step â€” very slow.
        This dataset loads both the image and a depth PNG that was pre-computed
        once by precompute_depth.py.  Training therefore never needs to run DPT.

    EXPECTED FILE STRUCTURE:
        image_directory/
            cat.jpg
            dog.png
            house.jpeg
            ...
        depth_directory/        <- created by precompute_depth.py
            cat.png             <- greyscale depth for cat.jpg
            dog.png
            house.png
            ...

    The depth PNG files must have the SAME STEM as the corresponding image
    (e.g. "cat.jpg" pairs with "cat.png").  The depth is stored as 8-bit
    greyscale (values 0-255), which this dataset normalises back to [0, 1]
    and replicates to 3 channels.

    Each __getitem__ returns:
        {
          "jpg"    : image tensor  [3, H, W]  in [-1, 1]   (standard training format)
          "depth"  : depth tensor  [3, H, W]  in [0, 1]    (all 3 channels identical)
          "caption": string caption for the image
        }
    """

    def __init__(
        self,
        image_directory: Path,
        depth_directory: Path,
        image_transform,             # torchvision Compose for the RGB image
        depth_size: int = 512,       # square size for depth map resizing
        caption_from_name: bool = False,
        caption_prefix: str = "",
    ):
        self.image_directory = image_directory
        self.depth_directory = depth_directory
        self.image_transform = image_transform
        self.caption_from_name = caption_from_name
        self.caption_prefix = caption_prefix

        # Depth transform: resize to exact square, then [0,255] -> [0,1] via ToTensor.
        # NOTE: We deliberately do NOT apply the Normalize(-0.5/0.5) step that the
        # image transform uses, because depth values must stay in [0, 1] — that is
        # what the mapper network expects (matches DepthEstimator output range).
        self.depth_transform = transforms.Compose([
            transforms.Resize((depth_size, depth_size)),
            transforms.ToTensor(),   # PIL uint8 [0,255] -> float [0,1]
        ])

        # Gather all image paths (sorted for reproducibility)
        valid_exts = (".jpg", ".jpeg", ".png")
        self.image_paths = sorted(
            [image_directory / f for f in os.listdir(image_directory)
             if f.lower().endswith(valid_exts)],
            key=sort_key,
        )

        # Warn early if any depth maps are missing so the user runs precompute_depth.py
        missing = [
            p.name for p in self.image_paths
            if not (depth_directory / (p.stem + ".png")).exists()
        ]
        if missing:
            print(
                f"[DepthImageFolderDataset] WARNING: {len(missing)} depth maps missing "
                f"in {depth_directory}.\n"
                f"  Run:  python precompute_depth.py "
                f"--input_dir {image_directory} --output_dir {depth_directory}\n"
                f"  First missing: {missing[:5]}"
            )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image_path = self.image_paths[idx]
        depth_path = self.depth_directory / (image_path.stem + ".png")
        txt_path   = image_path.with_suffix(".txt")

        # ---- Caption --------------------------------------------------------
        if self.caption_from_name:
            label = self.caption_prefix + image_path.stem.split("_")[0].replace("-", " ")
        else:
            try:
                with open(txt_path, "r") as f:
                    label = f.read()
            except Exception:
                label = ""

        # ---- RGB image (normalised to [-1, 1]) ------------------------------
        image = Image.open(image_path).convert("RGB")
        if self.image_transform:
            image = self.image_transform(image)

        # ---- Depth map (normalised to [0, 1], 3-channel) --------------------
        # The depth PNG was saved as greyscale by precompute_depth.py.
        # "L" mode = 8-bit single channel.  ToTensor converts to [1, H, W] float.
        depth = Image.open(depth_path).convert("L")
        depth = self.depth_transform(depth)              # [1, H, W] in [0, 1]
        depth = depth.repeat(3, 1, 1)                    # [3, H, W] â€” replicate channels
        # This matches the output format of DepthEstimator (midas.py) exactly.

        return {"jpg": image, "depth": depth, "caption": label}


class DepthImageDataModule:
    """
    Data module that pairs RGB images with their pre-computed depth maps.

    Wraps one or more DepthImageFolderDataset instances (one per image/depth
    directory pair) and exposes the standard train_dataloader / val_dataloader
    interface used by train_depth.py.

    USAGE IN CONFIG (configs/data/local_depth.yaml):
        _target_: src.data.local.DepthImageDataModule
        image_directories: ["data/train_images"]
        depth_directories: ["data/train_images_depth"]
        ...

    The image_directories and depth_directories lists must have equal length.
    If you have multiple training sets, list them in parallel:
        image_directories: ["data/set_A", "data/set_B"]
        depth_directories: ["data/set_A_depth", "data/set_B_depth"]
    They will be concatenated into a single dataset.
    """

    def __init__(
        self,
        image_directories: list,
        depth_directories: list,
        transform: list,              # Hydra-instantiated image transforms list
        size: int = 512,              # square size for depth maps (matches experiment size)
        val_image_directories: list = [],
        val_depth_directories: list = [],
        batch_size: int = 8,
        val_batch_size: int = 1,
        workers: int = 4,
        val_workers: int = 1,
        caption_from_name: bool = False,
        caption_prefix: str = "",
    ):
        assert len(image_directories) == len(depth_directories), (
            "image_directories and depth_directories must have the same number of entries. "
            f"Got {len(image_directories)} image dirs and {len(depth_directories)} depth dirs."
        )

        self.batch_size     = batch_size
        self.val_batch_size = val_batch_size
        self.workers        = workers
        self.val_workers    = val_workers

        # Project root: two levels up from this file (src/data/local.py -> project root)
        project_root  = Path(os.path.abspath(__file__)).parent.parent.parent
        image_tfm     = transforms.Compose(transform)

        # ---- Training datasets ----------------------------------------------
        train_ds_list = [
            DepthImageFolderDataset(
                image_directory=Path(project_root, img_dir),
                depth_directory=Path(project_root, dep_dir),
                image_transform=image_tfm,
                depth_size=size,
                caption_from_name=caption_from_name,
                caption_prefix=caption_prefix,
            )
            for img_dir, dep_dir in zip(image_directories, depth_directories)
        ]
        # ConcatDataset handles single-dataset case transparently
        self.train_dataset = ConcatDataset(train_ds_list)

        # ---- Validation datasets --------------------------------------------
        if val_image_directories and val_depth_directories:
            assert len(val_image_directories) == len(val_depth_directories), (
                "val_image_directories and val_depth_directories must match in length."
            )
            val_ds_list = [
                DepthImageFolderDataset(
                    image_directory=Path(project_root, img_dir),
                    depth_directory=Path(project_root, dep_dir),
                    image_transform=image_tfm,
                    depth_size=size,
                    caption_from_name=caption_from_name,
                    caption_prefix=caption_prefix,
                )
                for img_dir, dep_dir in zip(val_image_directories, val_depth_directories)
            ]
            self.val_dataset = ConcatDataset(val_ds_list)
        else:
            # Fall back to training set as validation if no val dirs given.
            # This is common for small custom datasets.
            self.val_dataset = self.train_dataset

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.val_workers,
        )


# ============================================================================ #
#  JSON-MANIFEST DATASET CLASSES                                               #
#  Use these when you want per-image prompts stored in a JSON file,            #
#  rather than deriving prompts from filenames.                                #
# ============================================================================ #

class DepthJsonDataset(Dataset):
    """
    Loads image + pre-computed depth map pairs described in a JSON manifest.

    JSON MANIFEST FORMAT (one list, each entry is one training sample):
        [
          {
            "raw_image_path": "data/images/cat.jpg",       <- path to RGB image
            "prompt":         "a cute cat on a chair",     <- text prompt for this image
            "depth_path":     "data/depths/cat.png"        <- depth PNG (added by precompute_depth.py)
          },
          ...
        ]

    Paths can be:
      - Absolute  â†’ used as-is
      - Relative  â†’ resolved relative to the project root first; if not found,
                    relative to the JSON file's directory.

    Each __getitem__ returns:
        {
          "jpg"    : image tensor  [3, H, W]  in [-1, 1]
          "depth"  : depth tensor  [3, H, W]  in [0, 1]
          "caption": text string (from "prompt" or "caption" key in JSON)
        }
    """

    def __init__(
        self,
        json_file: Path,
        image_transform,         # torchvision Compose for RGB images
        depth_size: int = 512,
        project_root: Path = None,
    ):
        self.json_file   = Path(json_file)
        self.json_dir    = self.json_file.parent
        self.project_root = Path(project_root) if project_root else self.json_dir

        with open(self.json_file, "r", encoding="utf-8") as f:
            self.items = json.load(f)

        self.image_transform = image_transform

        # Depth transform: resize to exact square + [0,255]→[0,1].  NO Normalize step —
        # depth values must stay in [0, 1] as the mapper network expects.
        self.depth_transform = transforms.Compose([
            transforms.Resize((depth_size, depth_size)),
            transforms.ToTensor(),
        ])

        # Pre-check all entries so missing files are caught early
        missing_img = missing_dep = 0
        for item in self.items:
            if not self._resolve(item["raw_image_path"]).exists():
                print(f"[DepthJsonDataset] WARN image not found: {item['raw_image_path']}")
                missing_img += 1
            if not item.get("depth_path", ""):
                missing_dep += 1
            elif not self._resolve(item["depth_path"]).exists():
                print(f"[DepthJsonDataset] WARN depth not found: {item['depth_path']}")
                missing_dep += 1
        if missing_dep > 0:
            print(
                f"[DepthJsonDataset] {missing_dep}/{len(self.items)} entries missing depth_path.\n"
                f"  Run: python precompute_depth.py --input_dir data/images --output_dir data/depths"
            )

    def _resolve(self, p: str) -> Path:
        """Resolve a path string: absolute â†’ as-is; relative â†’ try project_root, then json dir."""
        p = Path(p)
        if p.is_absolute():
            return p
        abs_p = self.project_root / p
        if abs_p.exists():
            return abs_p
        return self.json_dir / p

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]

        caption = item.get("prompt", "")

        # ---- RGB image -------------------------------------------------------
        image_path = self._resolve(item["raw_image_path"])
        image = Image.open(image_path).convert("RGB")
        if self.image_transform:
            image = self.image_transform(image)

        # ---- Depth map -------------------------------------------------------
        # Saved as 8-bit grayscale by precompute_depth.py.
        # ToTensor converts [0,255] -> [0,1], output is [1, H, W].
        # Replicated to 3 channels to match DepthEstimator's output format.
        depth_path = self._resolve(item["depth_path"])
        depth = Image.open(depth_path).convert("L")   # "L" = 8-bit grayscale
        depth = self.depth_transform(depth)            # [1, H, W] in [0, 1]
        depth = depth.repeat(3, 1, 1)                  # [3, H, W]

        return {"jpg": image, "depth": depth, "caption": caption}


class DepthJsonDataModule:
    """
    Data module that reads a JSON manifest for both training and optional validation.

    Use this when you have per-image prompts stored in a JSON file.
    Pairs with DepthJsonDataset and exposes the standard
    train_dataloader() / val_dataloader() interface expected by train_depth.py.

    CONFIG EXAMPLE (configs/data/local_depth_json.yaml):
        _target_: src.data.local.DepthJsonDataModule
        json_file: dataset.json         # path relative to project root
        val_json_file: null             # optional, falls back to train set
        size: 512
        batch_size: 8

    COMMAND-LINE OVERRIDE:
        python train_depth.py ... data.json_file=my_dataset.json
    """

    def __init__(
        self,
        json_file: str,
        transform: list,               # Hydra-instantiated image transforms
        size: int = 512,
        val_json_file: str = None,
        batch_size: int = 8,
        val_batch_size: int = 1,
        workers: int = 4,
        val_workers: int = 1,
    ):
        project_root = Path(os.path.abspath(__file__)).parent.parent.parent
        image_tfm    = transforms.Compose(transform)

        self.batch_size     = batch_size
        self.val_batch_size = val_batch_size
        self.workers        = workers
        self.val_workers    = val_workers

        self.train_dataset = DepthJsonDataset(
            json_file=Path(project_root, json_file),
            image_transform=image_tfm,
            depth_size=size,
            project_root=project_root,
        )

        if val_json_file:
            self.val_dataset = DepthJsonDataset(
                json_file=Path(project_root, val_json_file),
                image_transform=image_tfm,
                depth_size=size,
                project_root=project_root,
            )
        else:
            # No separate val set â†’ reuse training set (common for small datasets)
            self.val_dataset = self.train_dataset

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.val_workers,
        )

