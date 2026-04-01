"""Data curation for TRMC training.

This script provides utilities to load and process Wikipedia (WikiText),
Mathematics (OpenMath), Coding (The Stack), and Conversational (OpenAssistant)
datasets for TRMC training, along with Image datasets for vision encoding.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

class TRMCDataset(Dataset):
    """Real dataset for TRMC training using HuggingFace datasets."""

    def __init__(
        self,
        tokenizer,
        max_seq_len: int = 4096,
        num_samples: int = 10000,
        num_negatives: int = 3,
        include_vision: bool = True,
    ):
        """Initializes the TRMCDataset with real data.

        Args:
            tokenizer: Tokenizer to use for text encoding.
            max_seq_len (int): Maximum sequence length.
            num_samples (int): Number of samples to include.
            num_negatives (int): Number of negative examples per positive.
            include_vision (bool): Whether to include image data.
        """
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.num_samples = num_samples
        self.num_negatives = num_negatives
        self.include_vision = include_vision

        self.text_data = []
        self.image_data = []

        if not HAS_DATASETS:
             print("Warning: 'datasets' library not found. Falling back to simulated data.")
             self._generate_simulated_data()
             return

        print("Downloading and processing real datasets (Wiki, Math, Code, Oasst)...")
        # 1. Wikipedia (Knowledge)
        wiki = load_dataset("wikitext", "wikitext-103-v1", split="train", streaming=True)
        # 2. Math (Reasoning)
        math = load_dataset("nvidia/OpenMath-Instruct", split="train", streaming=True)
        # 3. Code (Logic)
        code = load_dataset("bigcode/the-stack-smol", split="train", streaming=True)
        # 4. Conversational (Dialogue)
        oasst = load_dataset("OpenAssistant/oasst1", split="train", streaming=True)

        # 5. Image (Vision)
        if include_vision:
            print("Downloading image dataset (COCO-style)...")
            vision = load_dataset("mbeukman/coco-captions-small", split="train", streaming=True)
            self.vision_iter = iter(vision)

        self.sources = [iter(wiki), iter(math), iter(code), iter(oasst)]
        self._load_real_data()

    def _load_real_data(self):
        """Loads and tokenizes data from real sources."""
        count = 0
        while count < self.num_samples:
            for source_iter in self.sources:
                try:
                    item = next(source_iter)
                    text = item.get("text") or item.get("instruction") or item.get("content") or ""
                    if len(text) < 50: continue # Skip very short texts

                    # Tokenize
                    tokens = self.tokenizer(
                        text,
                        truncation=True,
                        max_length=self.max_seq_len,
                        padding="max_length",
                        return_tensors="pt"
                    )
                    self.text_data.append(tokens["input_ids"].squeeze(0))

                    if self.include_vision:
                         try:
                             v_item = next(self.vision_iter)
                             # Process image (placeholder for actual image processing)
                             img = v_item["image"].convert("RGB").resize((224, 224))
                             img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
                             self.image_data.append(img_tensor)
                         except StopIteration:
                             # Re-start vision iterator if needed
                             pass

                    count += 1
                    if count >= self.num_samples: break
                except StopIteration:
                    continue

    def _generate_simulated_data(self):
        """Generates high-quality simulated data if real datasets are unavailable."""
        print(f"Simulating {self.num_samples} samples...")
        vocab_size = getattr(self.tokenizer, "vocab_size", 32000)
        for _ in range(self.num_samples):
            x = torch.randint(3, vocab_size, (self.max_seq_len,))
            self.text_data.append(x)
            if self.include_vision:
                self.image_data.append(torch.randn(3, 224, 224))

    def __len__(self) -> int:
        return len(self.text_data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        x = self.text_data[idx]
        # For TRMC training, we want (input, positive, negatives)
        # Positive is typically the next-token sequence (causal)
        y_pos = torch.roll(x, -1)
        y_pos[-1] = 0 # Padding token or EOS

        # Negatives: corrupted versions of the positive
        y_negs = []
        for _ in range(self.num_negatives):
            y_neg = y_pos.clone()
            # Corrupt 10% of tokens
            mask = torch.rand(y_neg.shape) < 0.1
            y_neg[mask] = torch.randint(3, getattr(self.tokenizer, "vocab_size", 32000), (mask.sum(),))
            y_negs.append(y_neg)

        img = self.image_data[idx] if self.include_vision else None

        return x, y_pos, torch.stack(y_negs), img

class MultiDomainTRMCDataset(Dataset):
    """Legacy compatibility class for simulated data."""
    def __init__(self, size: int = 5000, seq_len: int = 128, vocab_size: int = 32000, num_negatives: int = 3):
        self.size = size
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.num_negatives = num_negatives
        self.data = []

        print(f"Simulating Multi-Domain Data ({size} samples)...")
        for _ in range(size):
            x = torch.randint(3, vocab_size, (seq_len,))
            y_pos = torch.roll(x, -1)
            y_pos[-1] = torch.randint(3, vocab_size, (1,))
            y_negs = torch.stack([torch.randint(3, vocab_size, (seq_len,)) for _ in range(num_negatives)])
            self.data.append((x, y_pos, y_negs))

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.data[idx]

if __name__ == "__main__":
    # Test with dummy tokenizer
    class DummyTokenizer:
        def __init__(self): self.vocab_size = 32000
        def __call__(self, text, **kwargs):
            return {"input_ids": torch.randint(0, 32000, (1, kwargs.get("max_length", 128)))}

    tokenizer = DummyTokenizer()
    dataset = TRMCDataset(tokenizer, max_seq_len=128, num_samples=10, include_vision=True)
    print(f"Dataset length: {len(dataset)}")
    x, y_p, y_n, img = dataset[0]
    print(f"Shapes: x={x.shape}, y_pos={y_p.shape}, y_negs={y_n.shape}, img={img.shape if img is not None else 'None'}")
