"""
Create SCARED1500 evaluation subset from the SCARED dataset.

This script processes the SCARED dataset test set to create an evaluation subset
compatible with the posed_images dataset class used by other benchmarks.

Pairs are filtered to have overlap > 40%.

Usage:
    python -m gluefactory.scripts.create_scared1500 --input /path/to/scared --output ./data/scared1500
"""

import argparse
import json
import logging
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from .scared_utils import compute_overlap, load_scene_points

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Minimum overlap threshold
MIN_OVERLAP = 0.40


def parse_args():
    parser = argparse.ArgumentParser(description="Create SCARED1500 evaluation subset from SCARED dataset")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to SCARED dataset root (containing test/ folder)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./data/scared1500"),
        help="Output directory for scared1500 dataset",
    )
    parser.add_argument(
        "--num_pairs",
        type=int,
        default=1500,
        help="Number of image pairs to sample (default: 1500)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


def discover_frames(test_dir: Path) -> dict:
    """
    Discover all frames in the test directory.

    Returns:
        dict: {(dataset_name, keyframe_name): [frame_ids]}
    """
    keyframe_frames = {}

    # Find all dataset folders
    for dataset_dir in sorted(test_dir.iterdir()):
        if not dataset_dir.is_dir() or not dataset_dir.name.startswith("dataset_"):
            continue

        dataset_name = dataset_dir.name

        # Find all keyframe folders
        for keyframe_dir in sorted(dataset_dir.iterdir()):
            if not keyframe_dir.is_dir() or not keyframe_dir.name.startswith("keyframe_"):
                continue

            keyframe_name = keyframe_dir.name
            frame_data_dir = keyframe_dir / "data" / "frame_data"

            if not frame_data_dir.exists():
                logger.warning(f"No frame_data directory found in {keyframe_dir}")
                continue

            # Find all frame JSON files
            frame_ids = []
            for json_file in sorted(frame_data_dir.glob("frame_data*.json")):
                # Extract frame ID from filename (e.g., frame_data000123.json -> 000123)
                frame_id = json_file.stem.replace("frame_data", "")
                frame_ids.append(frame_id)

            if frame_ids:
                keyframe_frames[(dataset_name, keyframe_name)] = frame_ids
                logger.info(f"Found {len(frame_ids)} frames in {dataset_name}/{keyframe_name}")

    return keyframe_frames


def load_frame_metadata(test_dir: Path, dataset: str, keyframe: str, frame_id: str):
    """
    Load camera calibration and pose from frame JSON file.

    Returns:
        dict with keys: R, T, fx, fy, cx, cy, k1, k2, p1, p2, k3
    """
    json_path = test_dir / dataset / keyframe / "data" / "frame_data" / f"frame_data{frame_id}.json"

    with open(json_path) as f:
        data = json.load(f)

    # Camera intrinsics from KL
    KL = np.array(data["camera-calibration"]["KL"])
    fx, fy = KL[0][0], KL[1][1]
    cx, cy = KL[0][2], KL[1][2]

    # Distortion parameters from DL (all 5 for undistortion)
    DL = np.array(data["camera-calibration"]["DL"][0])
    k1, k2, p1, p2, k3 = DL[0], DL[1], DL[2], DL[3], DL[4]

    # Camera pose (4x4 matrix)
    pose = np.array(data["camera-pose"])
    R = pose[:3, :3]
    T = pose[:3, 3]

    return {
        "R": R,
        "T": T,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "k1": k1,
        "k2": k2,
        "p1": p1,
        "p2": p2,
        "k3": k3,
    }


def compute_pair_overlap(test_dir: Path, dataset: str, keyframe: str, f1: str, meta1: dict, meta2: dict) -> float:
    """
    Compute bidirectional overlap between two frames.

    Returns:
        float: Average of overlap from f1->f2 and f2->f1
    """
    # Load scene points
    pts1 = load_scene_points(test_dir, dataset, keyframe, f1)

    # Build camera matrix (same for both frames in a keyframe)
    K = np.array([[meta1["fx"], 0, meta1["cx"]], [0, meta1["fy"], meta1["cy"]], [0, 0, 1]])

    overlap_1to2 = compute_overlap(pts1, meta1, meta2, K)

    return overlap_1to2


def sample_pairs_with_overlap(test_dir: Path, keyframe_frames: dict, num_pairs: int, seed: int) -> tuple[list, dict]:
    """
    Sample pairs of frames filtered by overlap > MIN_OVERLAP.
    Uses lazy evaluation - generates random pairs on-demand until we have enough.

    Args:
        test_dir: Path to test directory
        keyframe_frames: {(dataset, keyframe): [frame_ids]}
        num_pairs: Number of pairs to sample
        seed: Random seed

    Returns:
        tuple: (list of sampled pairs with overlap, metadata cache)
    """
    rng = np.random.RandomState(seed)

    # Build list of keyframes (not all pairs!)
    keyframe_list = list(keyframe_frames.keys())

    # Count total possible pairs for logging and safety limit
    total_possible = sum(len(frames) * (len(frames) - 1) // 2 for frames in keyframe_frames.values())
    logger.info(f"Total possible pairs: {total_possible}")

    # Initialize tracking
    sampled_pairs = []
    seen_pairs = set()  # Track already-seen pairs to avoid duplicates

    # Metadata cache
    metadata_cache = {}

    def get_metadata(dataset: str, keyframe: str, frame_id: str) -> dict:
        key = (dataset, keyframe, frame_id)
        if key not in metadata_cache:
            metadata_cache[key] = load_frame_metadata(test_dir, dataset, keyframe, frame_id)
        return metadata_cache[key]

    def random_pair():
        """Generate a random pair from the same keyframe."""
        # Pick random keyframe
        key = keyframe_list[rng.randint(len(keyframe_list))]
        frames = keyframe_frames[key]

        # Pick two different frames
        idx1, idx2 = rng.choice(len(frames), size=2, replace=False)
        f1, f2 = frames[idx1], frames[idx2]

        # Ensure consistent ordering
        if f1 > f2:
            f1, f2 = f2, f1

        return key, f1, f2

    # Generate pairs until we have enough or we've tried too many times
    pairs_processed = 0
    max_attempts = total_possible * 2  # Safety limit

    with tqdm(total=num_pairs, desc="Sampling pairs") as pbar:
        while len(sampled_pairs) < num_pairs and pairs_processed < max_attempts:
            # Generate random pair
            (dataset, keyframe), f1, f2 = random_pair()
            pair_key = (dataset, keyframe, f1, f2)

            # Skip if already seen
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            pairs_processed += 1

            # Load metadata and compute overlap
            meta1 = get_metadata(dataset, keyframe, f1)
            meta2 = get_metadata(dataset, keyframe, f2)

            try:
                overlap = compute_pair_overlap(test_dir, dataset, keyframe, f1, meta1, meta2)
            except FileNotFoundError as e:
                logger.warning(f"Skipping pair due to missing file: {e}")
                continue

            # Check overlap threshold
            if overlap < MIN_OVERLAP:
                continue

            # Add pair
            sampled_pairs.append(((dataset, keyframe), f1, f2, overlap))
            pbar.update(1)

    logger.info(
        f"Processed {pairs_processed} unique pairs to find {len(sampled_pairs)} with overlap > {MIN_OVERLAP:.0%}"
    )

    return sampled_pairs, metadata_cache


def get_image_path(dataset: str, keyframe: str, frame_id: str) -> str:
    """Get the output image path for a frame (with _undistorted suffix)."""
    return f"{dataset}_undistorted/{keyframe}_frame_{frame_id}.png"


def get_source_image_path(test_dir: Path, dataset: str, keyframe: str, frame_id: str) -> Path:
    """Get the source image path in the SCARED dataset."""
    return test_dir / dataset / keyframe / "data" / "rgb_frames_left" / f"frame_{frame_id}.png"


def undistort_and_save(src_path: Path, dst_path: Path, meta: dict, width: int, height: int):
    """
    Load an image, undistort it, and save to destination.

    Returns:
        dict with new camera intrinsics (fx, fy, cx, cy)
    """
    # Load image
    img = cv2.imread(str(src_path))
    if img is None:
        raise ValueError(f"Failed to load image: {src_path}")

    # Build camera matrix and distortion coefficients
    K = np.array([[meta["fx"], 0, meta["cx"]], [0, meta["fy"], meta["cy"]], [0, 0, 1]], dtype=np.float64)
    dist_coeffs = np.array([meta["k1"], meta["k2"], meta["p1"], meta["p2"], meta["k3"]], dtype=np.float64)

    # Undistort the image
    undistorted = cv2.undistort(img, K, dist_coeffs, None)

    # Save undistorted image
    cv2.imwrite(str(dst_path), undistorted)

    # Return new intrinsics
    return {
        "fx": K[0, 0],
        "fy": K[1, 1],
        "cx": K[0, 2],
        "cy": K[1, 2],
    }


def main():
    args = parse_args()

    test_dir = args.input / "test"
    if not test_dir.exists():
        raise ValueError(f"Test directory not found: {test_dir}")

    output_dir = args.output
    images_dir = output_dir / "images"

    # Step 1: Discover all frames
    logger.info("Discovering frames in test directory...")
    keyframe_frames = discover_frames(test_dir)

    if not keyframe_frames:
        raise ValueError("No frames found in the test directory")

    total_frames = sum(len(frames) for frames in keyframe_frames.values())
    logger.info(f"Found {total_frames} total frames across {len(keyframe_frames)} keyframes")

    # Step 2: Sample pairs with overlap filtering (overlap > 40%)
    logger.info(f"Sampling {args.num_pairs} pairs with overlap > {MIN_OVERLAP:.0%} (seed {args.seed})...")
    sampled_pairs_with_overlap, image_metadata = sample_pairs_with_overlap(
        test_dir, keyframe_frames, args.num_pairs, args.seed
    )
    logger.info(f"Sampled {len(sampled_pairs_with_overlap)} pairs total")

    # Step 3: Collect unique images
    logger.info("Collecting unique images...")
    unique_images = set()
    for (dataset, keyframe), f1, f2, _overlap in sampled_pairs_with_overlap:
        unique_images.add((dataset, keyframe, f1))
        unique_images.add((dataset, keyframe, f2))

    logger.info(f"Found {len(unique_images)} unique images")

    # Image dimensions (fixed for SCARED)
    width, height = 1280, 1024

    # Step 4: Create output directory structure and undistort images
    logger.info("Creating output directory and undistorting images...")
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    # Store intrinsics after undistortion
    intrinsics = {}

    # Create dataset subdirectories and undistort images
    processed_count = 0
    for dataset, keyframe, frame_id in tqdm(unique_images, desc="Undistorting images"):
        # Create dataset directory with _undistorted suffix
        dataset_dir = images_dir / f"{dataset}_undistorted"
        dataset_dir.mkdir(exist_ok=True)

        # Source and destination paths
        src_path = get_source_image_path(test_dir, dataset, keyframe, frame_id)
        dst_filename = f"{keyframe}_frame_{frame_id}.png"
        dst_path = dataset_dir / dst_filename

        if not src_path.exists():
            logger.warning(f"Source image not found: {src_path}")
            continue

        # Undistort and save image
        key = (dataset, keyframe, frame_id)
        meta = image_metadata[key]
        intr = undistort_and_save(src_path, dst_path, meta, width, height)
        intrinsics[key] = intr
        processed_count += 1

    logger.info(f"Undistorted {processed_count} images")

    # Step 5: Generate pairs.txt (with overlap info as comment)
    logger.info("Generating pairs.txt...")
    pairs_path = output_dir / "pairs.txt"
    with open(pairs_path, "w") as f:
        for (dataset, keyframe), f1, f2, _overlap in sampled_pairs_with_overlap:
            img1 = get_image_path(dataset, keyframe, f1)
            img2 = get_image_path(dataset, keyframe, f2)
            f.write(f"{img1} {img2}\n")

    # Step 6: Generate views.txt
    logger.info("Generating views.txt...")
    views_path = output_dir / "views.txt"

    with open(views_path, "w") as f:
        for dataset, keyframe, frame_id in sorted(unique_images):
            img_path = get_image_path(dataset, keyframe, frame_id)
            key = (dataset, keyframe, frame_id)
            meta = image_metadata[key]
            intr = intrinsics[key]

            # Format: img_path R_flattened[9] T[3] PINHOLE width height fx fy cx cy
            R_flat = " ".join(map(str, meta["R"].flatten()))
            T_flat = " ".join(map(str, meta["T"]))

            line = (
                f"{img_path} {R_flat} {T_flat} "
                f"PINHOLE {width} {height} "
                f"{intr['fx']} {intr['fy']} {intr['cx']} {intr['cy']}\n"
            )
            f.write(line)

    # Step 7: Generate overlap statistics
    logger.info("Generating overlap_stats.txt...")
    stats_path = output_dir / "overlap_stats.txt"
    overlaps = [overlap for _, _, _, overlap in sampled_pairs_with_overlap]
    with open(stats_path, "w") as f:
        f.write("# SCARED1500 Overlap Statistics\n")
        f.write(f"# Seed: {args.seed}\n")
        f.write(f"# Num pairs: {args.num_pairs}\n")
        f.write(f"# Min overlap threshold: {MIN_OVERLAP:.0%}\n\n")
        f.write(f"# Overlap stats: min={min(overlaps):.4f}, max={max(overlaps):.4f}, ")
        f.write(f"mean={np.mean(overlaps):.4f}, median={np.median(overlaps):.4f}\n\n")
        f.write("# Per-pair overlap values:\n")
        f.write("# img1 img2 overlap\n")
        for (dataset, keyframe), f1, f2, overlap in sampled_pairs_with_overlap:
            img1 = get_image_path(dataset, keyframe, f1)
            img2 = get_image_path(dataset, keyframe, f2)
            f.write(f"{img1} {img2} {overlap:.4f}\n")

    logger.info(f"Done! Output saved to {output_dir}")
    logger.info(f"  - pairs.txt: {len(sampled_pairs_with_overlap)} pairs")
    logger.info(f"  - views.txt: {len(unique_images)} images")
    logger.info("  - overlap_stats.txt: overlap values for all pairs")
    logger.info(f"  - images/: {processed_count} images undistorted")


if __name__ == "__main__":
    main()
