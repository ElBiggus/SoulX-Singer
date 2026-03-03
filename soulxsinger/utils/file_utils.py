
"""
Description:
    This script contains a collection of functions designed to handle various
    file reading and writing operations. It provides utilities to read from files,
    write data to files, and perform file manipulation tasks.
"""

import os
import json
import torch

from tqdm import tqdm
from typing import List, Dict, Optional
from pathlib import Path
from omegaconf import OmegaConf, DictConfig


def write_jsonl(metadata: List[dict], file_path: Path):
    """Writes a list of dictionaries to a JSONL file.

    Args:
    metadata : List[dict]
        A list of dictionaries, each representing a piece of meta.
    file_path : Path
        The file path to save the JSONL file

    This function writes each dictionary in the list to a new line in the specified file.
    """
    with open(file_path, "w", encoding="utf-8") as f:
        for meta in tqdm(metadata, desc="writing jsonl"):
            # Convert dictionary to JSON string and write it to the file with a newline
            json_str = json.dumps(meta, ensure_ascii=False) + "\n"
            f.write(json_str)
    print(f"jsonl saved to {file_path}")


def read_jsonl(file_path: Path) -> List[dict]:
    """
    Reads a JSONL file and returns a list of dictionaries.

    Args:
    file_path : Path
        The path to the JSONL file to be read.

    Returns:
    List[dict]
        A list of dictionaries parsed from each line of the JSONL file.
    """
    metadata = []
    # Open the file for reading
    with open(file_path, "r", encoding="utf-8") as f:
        # Split the file into lines
        lines = f.read().splitlines()
    # Process each line
    for line in lines:
        # Convert JSON string back to dictionary and append to list
        meta = json.loads(line)
        metadata.append(meta)
    # Return the list of metadata
    return metadata


def load_config(config_path: Path) -> DictConfig:
    """Loads a configuration file and optionally merges it with a base configuration.

    Args:
    config_path (Path): Path to the configuration file.
    """
    # Load the initial configuration from the given path
    config = OmegaConf.load(config_path)

    # Check if there is a base configuration specified and merge if necessary
    if config.get("base_config", None) is not None:
        base_config = OmegaConf.load(config["base_config"])
        config = OmegaConf.merge(base_config, config)

    return config


def resolve_device(device: Optional[str] = None, gpu: Optional[int] = 0) -> str:
    """Resolve runtime device string.

    Priority:
    1) Explicit ``device`` argument (supports values like ``cuda:1`` / ``cpu`` / ``1``)
    2) ``gpu`` index from config mapped to ``cuda:{gpu}``
    3) CPU fallback when CUDA is unavailable
    """
    if not torch.cuda.is_available():
        return "cpu"

    cuda_count = torch.cuda.device_count()

    if device is not None and str(device).strip() != "":
        normalized = str(device).strip().lower()
        if normalized.isdigit():
            idx = int(normalized)
            if 0 <= idx < cuda_count:
                return f"cuda:{idx}"
            return "cuda:0"
        return str(device).strip()

    try:
        gpu_idx = int(gpu) if gpu is not None else 0
    except (TypeError, ValueError):
        gpu_idx = 0

    if gpu_idx < 0 or gpu_idx >= cuda_count:
        gpu_idx = 0

    return f"cuda:{gpu_idx}"


def resolve_device_from_config(config: DictConfig, device: Optional[str] = None) -> str:
    """Resolve device using optional override and ``config.gpu``."""
    gpu = config.get("gpu", 0) if config is not None else 0
    return resolve_device(device=device, gpu=gpu)