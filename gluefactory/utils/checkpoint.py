"""Resolve (and, if needed, download) trained matcher checkpoints.

Mirrors ``ripepp/utils/checkpoint.py`` so the RIPE++ + LightGlue matcher can
locate its weights automatically instead of requiring the user to pass a path.
"""

from pathlib import Path
from typing import Optional, Union

import torch

DEFAULT_CKPT = "ripe++lightglue.tar"

# Dummy placeholder URL - replace with the real host once available.
CKPT_URL_BASE = "https://datacloud.hhi.fraunhofer.de/public.php/dav/files/P6GTAKgejiaK9BC/"

CKPT_VARIANTS = {
    "default": DEFAULT_CKPT,  # ripe++lightglue.tar
}


def _checkpoint_cache_dir() -> Path:
    """Persistent per-user dir for downloaded checkpoints."""
    d = Path(torch.hub.get_dir()) / "checkpoints"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _resolve_or_download(filename: str) -> Path:
    """Locate a checkpoint file, downloading it into the cache if needed.

    Lookup order:
        1. A repo-local ``weights/<filename>`` (the cloned-repo / dev workflow).
        2. A previously cached copy under :func:`_checkpoint_cache_dir`.
        3. Otherwise download from the server into the cache and return it.
    """
    repo_local = Path("weights") / filename  # dev / cloned-repo workflow
    if repo_local.exists():
        print(f"Using existing checkpoint: {repo_local}")
        return repo_local

    cached = _checkpoint_cache_dir() / filename
    if cached.exists():
        print(f"Using cached checkpoint: {cached}")
        return cached

    print(f"Checkpoint '{filename}' not found. Downloading to {cached} ...")
    torch.hub.download_url_to_file(f"{CKPT_URL_BASE}{filename}", str(cached))
    print("Done.")
    return cached


def resolve_variant_checkpoint(variant: str = "default") -> Path:
    """Return the local path to a named weight variant, downloading if absent.

    Args:
        variant: One of the keys in ``CKPT_VARIANTS`` (e.g. "default").

    Returns:
        Path to the local checkpoint file (repo-local ``weights/`` if present,
        otherwise the per-user cache directory).

    Raises:
        ValueError: If ``variant`` is not a known variant name.
    """
    if variant not in CKPT_VARIANTS:
        raise ValueError(f"Unknown weight variant '{variant}'. Choose from {sorted(CKPT_VARIANTS)}.")
    return _resolve_or_download(CKPT_VARIANTS[variant])


def resolve_lightglue_weights(weights: Optional[Union[str, Path]] = None) -> Path:
    """Resolve the ``weights`` conf value to a concrete local checkpoint path.

    Accepts (in priority order):
        * ``None`` -> the "default" variant (auto-download).
        * An existing filesystem path -> used as-is (explicit user override).
        * A known variant name in ``CKPT_VARIANTS`` -> resolved/downloaded.
        * Any other string -> treated as a bare checkpoint filename.
    """
    if weights is None:
        return resolve_variant_checkpoint("default")

    if Path(weights).exists():
        return Path(weights)

    if str(weights) in CKPT_VARIANTS:
        return resolve_variant_checkpoint(str(weights))

    return _resolve_or_download(Path(weights).name)
