"""
Recreate the SCARED1500 evaluation subset from an existing manifest.

Unlike ``create_scared1500.py`` (which randomly samples image pairs), this script reads the
already-generated ``views.txt``/``pairs.txt``/``overlap_stats.txt`` from an existing scared1500
directory and reproduces the dataset *exactly* - skipping the (non-deterministic) overlap
sampling entirely.

The txt files are the durable manifest, but the undistorted ``images/`` folder is not (it is
git-ignored). Recreating the images requires the raw source pixels and the distortion
coefficients stored in the SCARED frame JSON files, so the source SCARED dataset is required
via ``--input``. The undistortion reuses the exact same helpers as ``create_scared1500.py`` so
the output is pixel-identical.

Usage:
    python -m gluefactory.scripts.recreate_scared1500 \
        --input /path/to/scared --source ./data/scared1500 \
        --output ./data/scared1500_recreated
"""

import argparse
import logging
import shutil
from pathlib import Path

from tqdm import tqdm

from .create_scared1500 import (
    get_source_image_path,
    load_frame_metadata,
    undistort_and_save,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Image dimensions (fixed for SCARED)
WIDTH, HEIGHT = 1280, 1024

# Manifest files copied verbatim from the source dataset directory.
MANIFEST_FILES = ("pairs.txt", "views.txt", "overlap_stats.txt")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Recreate the SCARED1500 evaluation subset from an existing manifest"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to SCARED dataset root (containing test/ folder)",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("./data/scared1500"),
        help="Existing scared1500 directory holding views.txt/pairs.txt/overlap_stats.txt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./data/scared1500_recreated"),
        help="Output directory for the recreated dataset",
    )
    return parser.parse_args()


def parse_image_path(img_path: str) -> tuple[str, str, str]:
    """Parse an output image path back into (dataset, keyframe, frame_id).

    Inverse of ``create_scared1500.get_image_path``, which produces paths like
    ``dataset_8_undistorted/keyframe_0_frame_000003.png``.
    """
    dataset_part, filename = img_path.split("/")

    # Strip the "_undistorted" suffix from the dataset folder name.
    dataset = dataset_part[: -len("_undistorted")]

    # filename: keyframe_<n>_frame_<id>.png
    stem = filename[: -len(".png")]
    keyframe_str, frame_id = stem.split("_frame_")

    return dataset, keyframe_str, frame_id


def read_manifest_images(views_path: Path) -> list[tuple[str, str, str]]:
    """Read views.txt and return the list of unique (dataset, keyframe, frame_id).

    views.txt has one line per unique image; the first whitespace-separated token is the
    image path.
    """
    images = []
    with open(views_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            img_path = line.split()[0]
            images.append(parse_image_path(img_path))
    return images


def main():
    args = parse_args()

    test_dir = args.input / "test"
    if not test_dir.exists():
        raise ValueError(f"Test directory not found: {test_dir}")

    # Validate manifest files exist in the source directory.
    for name in MANIFEST_FILES:
        if not (args.source / name).exists():
            raise ValueError(f"Manifest file not found: {args.source / name}")

    output_dir = args.output
    images_dir = output_dir / "images"

    # Step 1: Read the image manifest from views.txt.
    logger.info(f"Reading image manifest from {args.source / 'views.txt'}...")
    unique_images = read_manifest_images(args.source / "views.txt")
    logger.info(f"Found {len(unique_images)} unique images in manifest")

    # Step 2: Create output directory structure and recreate undistorted images.
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Recreating undistorted images from source...")
    processed_count = 0
    for dataset, keyframe, frame_id in tqdm(unique_images, desc="Undistorting images"):
        dataset_dir = images_dir / f"{dataset}_undistorted"
        dataset_dir.mkdir(exist_ok=True)

        src_path = get_source_image_path(test_dir, dataset, keyframe, frame_id)
        dst_path = dataset_dir / f"{keyframe}_frame_{frame_id}.png"

        if not src_path.exists():
            logger.warning(f"Source image not found: {src_path}")
            continue

        meta = load_frame_metadata(test_dir, dataset, keyframe, frame_id)
        undistort_and_save(src_path, dst_path, meta, WIDTH, HEIGHT)
        processed_count += 1

    logger.info(f"Recreated {processed_count} images")

    # Step 3: Copy the manifest files verbatim (guarantees an exact metadata match).
    logger.info("Copying manifest files...")
    for name in MANIFEST_FILES:
        shutil.copy2(args.source / name, output_dir / name)

    logger.info(f"Done! Recreated dataset saved to {output_dir}")
    logger.info(f"  - images/: {processed_count} images undistorted")
    logger.info(f"  - {', '.join(MANIFEST_FILES)}: copied from {args.source}")


if __name__ == "__main__":
    main()
