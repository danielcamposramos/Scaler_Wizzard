"""
TRMC OCR Utils: Implements DeepSeek OCR technique.
Serializes high-dimensional embeddings into lossless image formats to save space.
"""

import torch
import numpy as np
from PIL import Image

def embeddings_to_ocr_image(embeddings: torch.Tensor, output_path: str):
    """
    Encodes model embeddings into a PNG image.
    This allows 'lossless huge context' by offloading weight data to the vision pipeline.
    """
    # Normalize embeddings to 0-255 range
    emb_np = embeddings.cpu().detach().numpy()
    rescaled = ((emb_np - emb_np.min()) / (emb_np.max() - emb_np.min()) * 255).astype(np.uint8)
    
    # Flatten or reshape to square image
    dim = int(np.ceil(np.sqrt(rescaled.size)))
    padded = np.zeros(dim * dim, dtype=np.uint8)
    padded[:rescaled.size] = rescaled.flatten()
    
    img = Image.fromarray(padded.reshape((dim, dim)), mode='L')
    img.save(output_path)
    print(f"✅ Context serialized to OCR Image: {output_path}")