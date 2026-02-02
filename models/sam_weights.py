"""
STEP 2: DOWNLOAD SAM MODEL WEIGHTS
===================================

Rationale:
----------
SAM comes in three sizes: ViT-B (358M), ViT-L (1.2GB), ViT-H (2.4GB).
For materials microscopy with fine features (1-5 pixel grain boundaries and
ferroelastic lines), we need the highest resolution model.

Why ViT-L (Large):
- ViT-B: Too low resolution, misses thin lines
- ViT-L: Best balance of accuracy and speed for your RTX 5080 (16GB VRAM)
- ViT-H: Overkill, slower, and may not fit in memory with batch processing

Download and verify:
"""

import os
import urllib.request
from pathlib import Path

def download_sam_weights(model_type='vit_l', save_dir='models/sam_weights'):
    """
    Download SAM model weights from Meta's repository.
    
    Parameters:
    -----------
    model_type : str
        'vit_b', 'vit_l', or 'vit_h' (default: 'vit_l' for best balance)
    save_dir : str
        Directory to save model weights
        
    Returns:
    --------
    str : Path to downloaded weights
    
    Rationale:
    ----------
    Automated download ensures correct model version. We use ViT-L because:
    1. Your RTX 5080 has 16GB VRAM (sufficient for ViT-L)
    2. ViT-L provides better fine detail detection than ViT-B
    3. Processing speed: ~4-5 sec/image vs ~2-3 sec for ViT-B (acceptable)
    """
    
    model_urls = {
        'vit_b': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth',
        'vit_l': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth',
        'vit_h': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth'
    }
    
    if model_type not in model_urls:
        raise ValueError(f"model_type must be one of {list(model_urls.keys())}")
    
    # Create directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract filename from URL
    url = model_urls[model_type]
    filename = url.split('/')[-1]
    save_path = os.path.join(save_dir, filename)
    
    # Download if not exists
    if not os.path.exists(save_path):
        print(f"Downloading {model_type} weights (~1.2GB)...")
        urllib.request.urlretrieve(url, save_path)
        print(f"Downloaded to {save_path}")
    else:
        print(f"Weights already exist at {save_path}")
    
    return save_path

# Execute download
if __name__ == "__main__":
    weights_path = download_sam_weights(model_type='vit_l')
    print(f"SAM weights ready at: {weights_path}")