import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from ..models.cache_loader import CacheLoader
from ..settings import DATA_PATH
from ..utils.image import ImagePreprocessor, load_image
from ..utils.tools import fork_rng
from ..visualization.viz2d import plot_image_grid
from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)
scene_lists_path = Path(__file__).parent / "megadepth_scene_lists"


def sample_n(data, num, seed=None):
    if len(data) > num:
        selected = np.random.RandomState(seed).choice(len(data), num, replace=False)
        return data[selected]
    else:
        return data


class MegaDepth_DISK(BaseDataset):
    default_conf = {
        # paths
        "data_dir": "disk-data/megadepth/",
        "num_scenes": 135,  # total number of scenes in the dataset
        "overfit": False,
        "dataset_file": "dataset.json",  # pairs from DISK paper
        "load_training_pairs": {
            "do": False,
            "path_positive": None,
        },
        # Training
        "train_split": None,  # only for compatibility, disk is only used for training
        "train_num_per_scene": 500,
        # Validation
        "val_split": None,  # only for compatibility, disk is only used for training
        "val_num_per_scene": None,
        "val_pairs": None,
        # Test
        "test_split": None,  # only for compatibility, disk is only used for training
        "test_num_per_scene": None,
        "test_pairs": None,
        # image options
        "read_image": True,
        "grayscale": False,
        "preprocessing": ImagePreprocessor.default_conf,
        "reseed": False,
        "seed": 0,
        # features from cache
        "load_features": {
            "do": False,
            **CacheLoader.default_conf,
            "collate": False,
        },
    }

    def _init(self, conf):
        if not (DATA_PATH / conf.data_dir).exists():
            logger.info("Downloading the DISK dataset.")
            raise NotImplementedError("Dataset download not implemented.")
            # self.download()

    def get_dataset(self, split):
        assert split in ["train"], "Only 'train' split is available for DISK."
        return _PairDataset(self.conf, split)


class _PairDataset(torch.utils.data.Dataset):
    def __init__(self, conf, split, load_sample=True):
        self.root = DATA_PATH / conf.data_dir
        assert self.root.exists(), self.root
        self.split = split
        self.conf = conf

        if conf.load_features.do:
            self.feature_loader = CacheLoader(conf.load_features)

        self.preprocessor = ImagePreprocessor(conf.preprocessing)

        if not conf.load_training_pairs.do:
            logger.info("Creating training pairs")
            self.positive_pairs = self._create_pairs()
        else:
            logger.info("Loading training pairs from disk")
            logger.info(f"Positive pairs from {conf.load_training_pairs.path_positive}")
            with np.load(DATA_PATH / conf.load_training_pairs.path_positive) as f:
                self.positive_pairs = f["arr_0"]
            logger.info(f"Loaded {len(self.positive_pairs)} positive pairs.")

        if load_sample:
            self.sample_new_items(conf.seed)
            assert len(self.sampled_positive_pairs) > 0

    def _create_pairs(self):
        json_path = self.root / self.conf.dataset_file
        assert json_path.exists(), f"Cannot find dataset file {json_path}."
        with open(json_path) as json_file:
            json_data = json.load(json_file)

        scenes = []
        for _idx, scene in enumerate(json_data):
            scenes.append(Scene(self.root, json_data[scene]))

        positive_pairs = []
        for scene in tqdm(scenes):
            for tuple_ in scene.tuples:
                positive_pairs.append(
                    (
                        str(scene.image_path / scene.image_names[tuple_[0]]),
                        str(scene.image_path / scene.image_names[tuple_[1]]),
                    )
                )
                positive_pairs.append(
                    (
                        str(scene.image_path / scene.image_names[tuple_[1]]),
                        str(scene.image_path / scene.image_names[tuple_[0]]),
                    )
                )

        return positive_pairs

    def sample_new_items(self, seed):
        logger.info("Sampling new %s data with seed %d.", self.split, seed)

        self.sampled_positive_pairs = sample_n(
            np.array(self.positive_pairs), self.conf.train_num_per_scene * self.conf.num_scenes, seed
        )

        logger.info(f"Sampled {len(self.sampled_positive_pairs)} positive pairs.")

    def _read_view(self, path, scene):
        # read image
        if self.conf.read_image:
            img = load_image(path, self.conf.grayscale)
        else:
            size = PIL.Image.open(path).size[::-1]
            img = torch.zeros([3 - 2 * int(self.conf.grayscale), size[0], size[1]]).float()

        name = path.name

        data = self.preprocessor(img)

        data = {
            "name": name,
            "scene": scene,
            **data,
        }

        if self.conf.load_features.do:
            features = self.feature_loader({k: [v] for k, v in data.items()})
            data = {"cache": features, **data}

        # if self.conf.load_features.do:
        #     path_feature_file = (DATA_PATH / self.conf.load_features.path / path.relative_to(self.root)).with_suffix(
        #         ".npz"
        #     )

        #     features = {}
        #     with np.load(path_feature_file) as features_npz:
        #         for k, v in features_npz.items():
        #             features[k] = torch.from_numpy(v).to(dtype=self.numeric_dtype)

        #     data = {"cache": features, **data}
        return data

    def __getitem__(self, idx):
        if self.conf.reseed:
            with fork_rng(self.conf.seed + idx, False):
                return self.getitem(idx)
        else:
            return self.getitem(idx)

    def getitem(self, idx):
        path0, path1 = self.sampled_positive_pairs[idx]

        i_scene_0 = path0[6:11]
        i_scene_1 = path1[6:11]

        path0 = self.root / path0
        path1 = self.root / path1

        data0 = self._read_view(path0, i_scene_0)
        data1 = self._read_view(path1, i_scene_1)
        data = {
            "view0": data0,
            "view1": data1,
        }

        data["label"] = True
        data["name"] = f"{path0}_{path1}"

        return data

    def __len__(self):
        return len(self.sampled_positive_pairs)


class Scene:
    def __init__(self, root_path, scene_data) -> None:
        self.root_path = root_path
        self.image_path = Path(scene_data["image_path"])
        self.image_names = scene_data["images"]

        self.tuples = scene_data["tuples"]

    def __len__(self) -> int:
        return len(self.tuples)

    def __getitem__(self, idx: int):
        if self.overfit:
            idx_1, idx_2 = 0, 1  # always the same pair in overfit mode
        else:
            idx_1, idx_2 = self.random_state_tuple_sampling.choice([0, 1, 2], 2, replace=False)

        idx_1 = self.tuples[idx][idx_1]
        idx_2 = self.tuples[idx][idx_2]

        path_image_1 = self.root_path / self.image_path / self.image_names[idx_1]
        path_image_2 = self.root_path / self.image_path / self.image_names[idx_2]

        return path_image_1, path_image_2


def visualize(args):
    conf = {
        "train_num_per_scene": 5,
        "batch_size": 1,
        "num_workers": 0,
        "prefetch_factor": None,
        "val_num_per_scene": None,
    }
    conf = OmegaConf.merge(conf, OmegaConf.from_cli(args.dotlist))
    dataset = MegaDepth_DISK(conf)
    loader = dataset.get_data_loader(args.split)
    logger.info(f"The dataset has {len(loader)} elements.")

    with fork_rng(seed=1):
        images = []
        for _, data in zip(range(args.num_items), loader, strict=False):
            images.append([data[f"view{i}"]["image"][0].permute(1, 2, 0) for i in range(2)])

    _ = plot_image_grid(images, dpi=args.dpi)
    # for i in range(len(images)):
    #     plot_heatmaps(depths[i], axes=axes[i])
    # plt.show()
    # save instead of showing
    plt.savefig("megadepth_dataset_visualization.png", dpi=args.dpi)


if __name__ == "__main__":
    from .. import logger  # overwrite the logger

    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--num_items", type=int, default=6)
    parser.add_argument("--dpi", type=int, default=100)
    parser.add_argument("dotlist", nargs="*")
    args = parser.parse_intermixed_args()
    visualize(args)
