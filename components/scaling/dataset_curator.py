"""Dataset curation and ingestion for Scaler Wizard.

This module handles the loading of real-world datasets from Hugging Face 
and prepares them for the TRMC (Recursive/MoE) pipeline, including mixing
multiple datasets for a unified training stream.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any
from datasets import load_dataset # type: ignore
from torch.utils.data import Dataset, ConcatDataset # type: ignore
import random

class DatasetCurator:
    """Curates and filters datasets for specific TRMC capabilities."""

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir

    def ingest_hf_dataset(self, path: str, split: str = "train", limit: int = 1000, name: Optional[str] = None):
        """
        Loads a dataset from Hugging Face and applies standard Scaler Wizard formatting.
        """
        print(f"Ingesting real-world dataset: {path}...")
        try:
            # Added 'name' parameter for datasets like Wikipedia that require specific configs
            dataset = load_dataset(path, name=name, split=f"{split}[:{limit}]", cache_dir=self.cache_dir, trust_remote_code=True)
            return dataset
        except Exception as e:
            print(f"Error loading dataset {path}: {e}")
            return None

    def get_math_stack(self, limit: int = 5000):
        """Pulls the highest quality math reasoning traces (NuminaMath)."""
        return self.ingest_hf_dataset("AI-MO/NuminaMath-CoT", limit=limit)

    def get_wikipedia_core(self, limit: int = 10000):
        """Pulls the 2022 English Wikipedia (Gold Standard)."""
        return self.ingest_hf_dataset("wikipedia", name="20220301.en", limit=limit)

    def get_game_logic_stack(self, limit: int = 5000):
        """
        Pulls Chess PGN logic and Text-Adventures.
        Perfect for training Recursive MoE experts in step-by-step logic.
        """
        chess = self.ingest_hf_dataset("laion/pgn-chess-proft", limit=limit // 2)
        adventure = self.ingest_hf_dataset("facebook/light_dialog", limit=limit // 2)
        return {"chess_logic": chess, "text_adventure": adventure}

    def prepare_trmc_mixture(self, dataset_paths: List[str]):
        """
        Combines multiple datasets (e.g. math + code + chat) to train a Sparse MoE model.
        """
        # Logic to interleave datasets for expert specialization
        mixtures = []
        for p in dataset_paths:
            data = self.ingest_hf_dataset(p)
            if data:
                mixtures.append(data)
        return mixtures

    def format_contrastive_pairs(self, dataset):
        """
        Formats DPO/ORPO datasets into contrastive pairs for TRMC latent alignment.
        Expects columns: 'prompt', 'chosen', 'rejected'.
        """
        def transform(example):
            return {
                "input": example["prompt"],
                "positive": example["chosen"],
                "negative": example["rejected"]
            }
        return dataset.map(transform)

    def create_mixed_dataset(self, dataset_configs: Dict[str, Dict[str, Any]]) -> "MixedDataset":
        """
        Loads and mixes multiple datasets based on provided configurations.
        dataset_configs: {
            "dataset_name_1": {"path": "hf_path_1", "limit": 1000, "name": None, "weight": 0.5},
            "dataset_name_2": {"path": "hf_path_2", "limit": 500, "weight": 0.3},
            ...
        }
        """
        loaded_datasets = {}
        weights = {}
        for ds_key, config in dataset_configs.items():
            path = config["path"]
            limit = config.get("limit", 1000)
            name = config.get("name", None)
            weight = config.get("weight", 1.0) # Default weight if not specified

            dataset = self.ingest_hf_dataset(path, limit=limit, name=name)
            if dataset:
                loaded_datasets[ds_key] = dataset["train"] # Assuming 'train' split
                weights[ds_key] = weight
        
        if not loaded_datasets:
            raise ValueError("No datasets were successfully loaded for mixing.")

        return MixedDataset(loaded_datasets, weights)

class MixedDataset(Dataset):
    """
    A dataset that mixes samples from multiple Hugging Face datasets.
    Can be configured with weights for each dataset.
    """
    def __init__(self, datasets: Dict[str, Dataset], weights: Dict[str, float]):
        self.datasets = datasets
        self.dataset_names = list(datasets.keys())
        
        # Create a list of (dataset_name, index_within_dataset) tuples for sampling
        self.sample_indices = []
        for name in self.dataset_names:
            num_samples = int(len(datasets[name]) * weights[name] * len(datasets) / sum(weights.values())) # Scale to roughly maintain proportions
            self.sample_indices.extend([(name, i % len(datasets[name])) for i in range(num_samples)])
        
        random.shuffle(self.sample_indices)

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        dataset_name, item_idx = self.sample_indices[idx]
        item = self.datasets[dataset_name][item_idx]
        
        # Standardize keys: 'text' for positive, 'negative_text' for negative signal
        # This handles DPO/ORPO style datasets mixed with Wikipedia
        return {
            "text": item.get("text") or item.get("chosen") or item.get("prompt", ""),
            "negative_text": item.get("rejected") or None
        }

    def get_example_by_dataset(self, dataset_name, idx):
        """Allows direct access to an example from a specific dataset."""
        return self.datasets[dataset_name][idx]