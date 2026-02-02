
"""
Fine-tuning script for MatSAM (Materials Segment Anything Model) on SEM images.

This script fine-tunes the Segment Anything Model (SAM) specifically for microstructure
segmentation in Scanning Electron Microscopy (SEM) images. It focuses on adapting the
mask decoder while keeping the image encoder frozen to preserve learned features.

Features:
    - Early stopping with configurable patience
    - Geometric data augmentation (flips, rotations)
    - Intensity data augmentation (brightness, contrast)
    - Multi-worker data loading with memory pinning
    - Reflection padding to avoid border artifacts
    - Automatic train/validation splitting
    - Loss curve visualization and preview generation

Typical usage:
    python finetuning/train_sam.py --train_data "data/tiled_images" \
                                   --mask_data "data/tiled_masks" \
                                   --epochs 50 --batch_size 4 \
                                   --val_split 0.2 --patience 5 \
                                   --augment --output_dir "finetuning/runs/exp1"
"""

import os
import sys
import torch
import random

# Add project root to path so 'core' can be found
sys.path.append(os.getcwd())

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import cv2
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry

class SAMDataset(Dataset):
    """PyTorch Dataset for SAM fine-tuning on image-mask pairs.
    
    This dataset handles loading, preprocessing, padding, and optional augmentation
    of SEM images and their corresponding grain boundary masks for SAM training.
    
    Args:
        image_dir (str or Path): Directory containing input SEM images (.png, .jpg, .tif).
        mask_dir (str or Path): Directory containing binary masks (.png, .jpg, .tif).
        processor (ResizeLongestSide): SAM's image preprocessing transform.
        size (int, optional): Target size for padding (default: 1024).
        augment (bool, optional): Enable geometric augmentations (default: False).
    
    Attributes:
        image_paths (list): Sorted list of matched image file paths.
        mask_paths (list): Sorted list of matched mask file paths.
    """
    
    def __init__(self, image_dir, mask_dir, processor, size=1024, augment=False):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.processor = processor
        self.size = size
        self.augment = augment
        
        self.image_paths = sorted(list(self.image_dir.glob("*.png")) + list(self.image_dir.glob("*.jpg")) + list(self.image_dir.glob("*.tif")))
        self.mask_paths = sorted(list(self.mask_dir.glob("*.png")) + list(self.mask_dir.glob("*.jpg")) + list(self.mask_dir.glob("*.tif")))
        
        if len(self.image_paths) != len(self.mask_paths):
            print(f"[WARNING] Mismatch: {len(self.image_paths)} images vs {len(self.mask_paths)} masks.")
            # Use only existing pairs
            img_names = {p.stem: p for p in self.image_paths}
            mask_names = {p.stem: p for p in self.mask_paths}
            common_stems = sorted(list(set(img_names.keys()) & set(mask_names.keys())))
            self.image_paths = [img_names[s] for s in common_stems]
            self.mask_paths = [mask_names[s] for s in common_stems]
            print(f"[INFO] Using {len(self.image_paths)} matched image-mask pairs.")
        else:
            print(f"[INFO] Found {len(self.image_paths)} image-mask pairs.")

    def __len__(self):
        """Returns the total number of image-mask pairs in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Load and preprocess a single image-mask pair.
        
        Processing pipeline:
            1. Load image and mask from disk
            2. Resize longest side to 1024px (SAM standard)
            3. Pad to exactly 1024x1024 using reflection/edge modes
            4. Apply geometric augmentations if enabled (synchronized)
            5. Convert to tensors and resize mask to 256x256 (decoder input size)
        
        Args:
            idx (int): Index of the sample to retrieve.
        
        Returns:
            dict: Dictionary containing:
                - 'image': Preprocessed image tensor (3, 1024, 1024)
                - 'mask': Binary mask tensor (1, 256, 256)
                - 'original_size': Original image dimensions (H, W)
        """
        img_path = str(self.image_paths[idx])
        mask_path = str(self.mask_paths[idx])
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32) # Binary mask
        
        # 1. Resize everything so longest side is 1024
        input_image = self.processor.apply_image(image)
        h_new, w_new = input_image.shape[:2]
        
        # Resize mask to the same new dimensions
        mask_resized = cv2.resize(mask, (w_new, h_new), interpolation=cv2.INTER_NEAREST)
        
        # 2. Pad to exactly 1024x1024 to allow batching
        # Image padding
        pad_h = self.size - h_new
        pad_w = self.size - w_new
        
        # Pad image (H, W, 3) -> (1024, 1024, 3) using reflection to avoid black border artifacts
        input_image_padded = np.pad(input_image, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
        
        # Pad mask (H, W) -> (1024, 1024) using edge mode (reflection would create false boundaries)
        mask_padded = np.pad(mask_resized, ((0, pad_h), (0, pad_w)), mode='edge')
        
        # 3. Apply augmentations (geometric only, synchronized for image+mask)
        if self.augment:
            # Horizontal flip (50% chance)
            if random.random() > 0.5:
                input_image_padded = np.fliplr(input_image_padded).copy()
                mask_padded = np.fliplr(mask_padded).copy()
            
            # Vertical flip (50% chance)
            if random.random() > 0.5:
                input_image_padded = np.flipud(input_image_padded).copy()
                mask_padded = np.flipud(mask_padded).copy()
            
            # Random 90-degree rotation (0, 90, 180, or 270 degrees)
            k = random.randint(0, 3)
            if k > 0:
                input_image_padded = np.rot90(input_image_padded, k).copy()
                mask_padded = np.rot90(mask_padded, k).copy()

            # 4. Brightness & Contrast Augmentation (50% chance)
            if random.random() > 0.5:
                # alpha: contrast [0.8, 1.2], beta: brightness [-30, 30]
                alpha = random.uniform(0.8, 1.2)
                beta = random.uniform(-30, 30)
                input_image_padded = np.clip(input_image_padded.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        
        # 3. Convert to tensors
        input_image_torch = torch.as_tensor(input_image_padded).permute(2, 0, 1).contiguous().float()
        
        # Resize mask to 256x256 (SAM internal decoder resolution)
        # Note: We resize the PADDED 1024 mask to 256 to maintain coordinate alignment
        mask_256 = cv2.resize(mask_padded, (256, 256), interpolation=cv2.INTER_NEAREST)
        
        return {
            "image": input_image_torch,
            "mask": torch.as_tensor(mask_256).unsqueeze(0),
            "original_size": torch.as_tensor([h, w])
        }

class MatSAMTrainer:
    """Trainer class for fine-tuning SAM on microstructure segmentation.
    
    This trainer handles model initialization, checkpoint management, training loops,
    validation, early stopping, and visualization of training progress.
    
    Only the mask decoder is fine-tuned while the image encoder and prompt encoder
    remain frozen to leverage SAM's pre-trained visual features.
    
    Args:
        args (Namespace): Parsed command-line arguments containing:
            - model_type: SAM architecture variant (e.g., 'vit_l')
            - checkpoint: Path to pre-trained SAM weights
            - output_dir: Directory for saving checkpoints and plots
            - lr: Learning rate for optimizer
            - epochs: Maximum number of training epochs
            - patience: Early stopping patience
    
    Attributes:
        sam (SamModel): The SAM model instance.
        optimizer (Adam): Optimizer for mask decoder parameters.
        criterion (BCEWithLogitsLoss): Binary cross-entropy loss.
        history (dict): Training and validation loss history.
        start_epoch (int): Starting epoch (1 or resumed epoch).
    """
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading model: {args.model_type} from {args.checkpoint}")
        self.sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
        self.sam.to(self.device)
        
        # Fine-tune only mask decoder
        for param in self.sam.image_encoder.parameters():
            param.requires_grad = False
        for param in self.sam.prompt_encoder.parameters():
            param.requires_grad = False
        for param in self.sam.mask_decoder.parameters():
            param.requires_grad = True
            
        self.optimizer = optim.Adam(self.sam.mask_decoder.parameters(), lr=args.lr)
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Prep directories
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plot_dir = self.output_dir / "plots"
        self.plot_dir.mkdir(exist_ok=True)
        
        # Load existing checkpoint if resuming
        self.start_epoch = 1
        self.history = {"train_loss": [], "val_loss": []}
        
        checkpoint_path = self.output_dir / "best_model.pth"
        history_path = self.output_dir / "history.json"
        
        if checkpoint_path.exists() and history_path.exists():
            print(f"Found existing checkpoint at {checkpoint_path}")
            print(f"Resuming training from previous run...")
            
            # Load model weights
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            self.sam.load_state_dict(state_dict)
            
            # Load history
            with open(history_path, "r") as f:
                self.history = json.load(f)
            
            self.start_epoch = len(self.history["train_loss"]) + 1
            print(f"Resuming from epoch {self.start_epoch}")
            print(f"Previous best train loss: {min(self.history['train_loss']):.4f}")
            if self.history["val_loss"]:
                print(f"Previous best val loss: {min(self.history['val_loss']):.4f}")

    def save_preview(self, epoch, image_torch, pred_mask, gt_mask):
        """Generate and save a visual comparison of predictions vs ground truth.
        
        Creates a 3-panel figure showing the original image, ground truth mask,
        and model prediction for the first sample of each epoch.
        
        Args:
            epoch (int): Current epoch number.
            image_torch (Tensor): Input image tensor (3, H, W).
            pred_mask (Tensor): Predicted mask logits (1, 256, 256).
            gt_mask (Tensor): Ground truth binary mask (1, 256, 256).
        """
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        # Convert torch (3, H, W) to numpy (H, W, 3) for plotting
        # Note: image_torch is already preprocessed by SAM (resized/padded)
        img_np = image_torch.cpu().permute(1, 2, 0).numpy()
        # Denormalize briefly for visualization if needed, but SAM preprocess is mostly just scaling
        # For simplicity, we just clip and show
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
        plt.imshow(img_np)
        plt.title("Original")
        
        plt.subplot(1, 3, 2)
        plt.imshow(gt_mask.cpu().squeeze(), cmap='gray')
        plt.title("Ground Truth")
        
        plt.subplot(1, 3, 3)
        plt.imshow(torch.sigmoid(pred_mask).cpu().detach().squeeze() > 0.5, cmap='jet')
        plt.title("Prediction")
        
        plt.savefig(self.plot_dir / f"epoch_{epoch}_preview.png")
        plt.close()

    def plot_history(self):
        """Generate and save training/validation loss curves.
        
        Creates a line plot showing loss progression over epochs for both
        training and validation sets (if validation is enabled).
        
        Output:
            Saves 'loss_curve.png' to the output directory.
        """
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(self.history["train_loss"]) + 1)
        plt.plot(epochs, self.history["train_loss"], 'o-', color='orange', label='Train Loss')
        
        if self.history.get("val_loss"):
            plt.plot(epochs, self.history["val_loss"], 'o-', color='blue', label='Val Loss')
            
        plt.title("Training Loss Over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.legend()
        plt.savefig(self.output_dir / "loss_curve.png")
        plt.close()

    def train(self, train_loader, val_loader=None):
        """Execute the training loop with validation and early stopping.
        
        Training process:
            1. Forward pass through frozen image encoder
            2. Train mask decoder with BCE loss
            3. Validate on held-out set (if provided)
            4. Save best checkpoint based on validation loss
            5. Apply early stopping if no improvement
        
        Args:
            train_loader (DataLoader): Training data loader.
            val_loader (DataLoader, optional): Validation data loader.
        
        Side effects:
            - Saves 'best_model.pth' when validation loss improves
            - Saves 'history.json' after each epoch
            - Generates 'loss_curve.png' and preview images
        """
        print(f"Starting training from epoch {self.start_epoch} to {self.args.epochs}...")
        print(f"Early stopping patience: {self.args.patience} epochs")
        
        best_val_loss = min(self.history["val_loss"]) if self.history["val_loss"] else float('inf')
        patience_counter = 0
        
        for epoch in range(self.start_epoch, self.args.epochs + 1):
            # --- Training Phase ---
            self.sam.mask_decoder.train()
            epoch_train_loss = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{self.args.epochs} [Train]")
            for i, batch in enumerate(pbar):
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                with torch.no_grad():
                    # CRITICAL: Preprocess images to match inference pipeline
                    images = self.sam.preprocess(images)
                    image_embedding = self.sam.image_encoder(images)
                
                sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
                    points=None, boxes=None, masks=None
                )
                
                low_res_masks, iou_predictions = self.sam.mask_decoder(
                    image_embeddings=image_embedding,
                    image_pe=self.sam.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )
                
                loss = self.criterion(low_res_masks, masks)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_train_loss += loss.item()
                pbar.set_postfix({"loss": loss.item()})
                
                if i == 0:
                    # Save a preview of the first batch's first item
                    # Note: we use the raw 'images' tensor before sam.preprocess for cleaner visualization
                    self.save_preview(epoch, batch['image'][0], low_res_masks[0], masks[0])

            avg_train_loss = epoch_train_loss / len(train_loader)
            self.history["train_loss"].append(avg_train_loss)

            # --- Validation Phase ---
            avg_val_loss = None
            if val_loader:
                self.sam.mask_decoder.eval()
                epoch_val_loss = 0
                with torch.no_grad():
                    vbar = tqdm(val_loader, desc=f"Epoch {epoch}/{self.args.epochs} [Val]")
                    for batch in vbar:
                        images = batch['image'].to(self.device)
                        masks = batch['mask'].to(self.device)
                        
                        # Preprocess images
                        images = self.sam.preprocess(images)
                        image_embedding = self.sam.image_encoder(images)
                        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
                            points=None, boxes=None, masks=None
                        )
                        low_res_masks, _ = self.sam.mask_decoder(
                            image_embeddings=image_embedding,
                            image_pe=self.sam.prompt_encoder.get_dense_pe(),
                            sparse_prompt_embeddings=sparse_embeddings,
                            dense_prompt_embeddings=dense_embeddings,
                            multimask_output=False,
                        )
                        loss = self.criterion(low_res_masks, masks)
                        epoch_val_loss += loss.item()
                        vbar.set_postfix({"loss": loss.item()})
                
                avg_val_loss = epoch_val_loss / len(val_loader)
                self.history["val_loss"].append(avg_val_loss)
                print(f"Epoch {epoch} Summary: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
            else:
                print(f"Epoch {epoch} Summary: Train Loss: {avg_train_loss:.4f}")

            # --- Save Checkpoint ---
            # Compare based on val_loss if available, otherwise train_loss
            monitor_loss = avg_val_loss if avg_val_loss is not None else avg_train_loss
            if monitor_loss < best_val_loss:
                best_val_loss = monitor_loss
                patience_counter = 0
                torch.save(self.sam.state_dict(), self.output_dir / "best_model.pth")
                print(f"  New best loss: {best_val_loss:.4f}, saved checkpoint.")
            else:
                patience_counter += 1
                print(f"  No improvement. Patience: {patience_counter}/{self.args.patience}")
            
            with open(self.output_dir / "history.json", "w") as f:
                json.dump(self.history, f)
            
            self.plot_history()
            
            # Early stopping check
            if patience_counter >= self.args.patience:
                print(f"\n🛑 Early stopping triggered after {epoch} epochs (no improvement for {self.args.patience} epochs).")
                break

        print(f"Training Complete. Results saved to {self.output_dir}")

def parse_args():
    """Parse command-line arguments for training configuration.
    
    Returns:
        Namespace: Parsed arguments with training hyperparameters and paths.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, required=True, help="Folder with tiled images")
    parser.add_argument("--mask_data", type=str, help="Folder with masks")
    parser.add_argument("--output_dir", type=str, default="finetuning/runs/exp1")
    parser.add_argument("--checkpoint", type=str, default="models/sam_weights/sam_vit_l_0b3195.pth")
    parser.add_argument("--model_type", type=str, default="vit_l")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0, help="Number of data loading workers (use 0 on Windows to avoid deadlocks)")
    parser.add_argument("--val_split", type=float, default=0.1, help="Fraction of data for validation (0.0 to 1.0)")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (epochs without improvement)")
    parser.add_argument("--augment", action="store_true", help="Enable geometric augmentations (flips, rotations)")
    return parser.parse_args()

def main():
    """Main entry point for SAM fine-tuning.
    
    Workflow:
        1. Parse arguments and validate paths
        2. Create dataset with optional augmentation
        3. Split into train/validation sets
        4. Initialize multi-worker data loaders
        5. Create trainer and start training loop
    """
    args = parse_args()
    
    # Resolve image/mask paths
    img_dir = Path(args.train_data)
    mask_dir = Path(args.mask_data) if args.mask_data else img_dir / "masks"

    if not mask_dir.exists():
        print(f"❌ Error: Mask directory {mask_dir} not found.")
        return

    # SAM uses specific resizing
    from segment_anything.utils.transforms import ResizeLongestSide
    processor = ResizeLongestSide(1024)
    
    full_dataset = SAMDataset(img_dir, mask_dir, processor, augment=args.augment)
    if len(full_dataset) == 0:
        print("❌ Error: No image-mask pairs found. Check your folders.")
        return
    
    if args.augment:
        print("[INFO] Augmentation ENABLED: H-Flip, V-Flip, 90° Rotations, Brightness/Contrast")

    # Train/Val Split
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    
    if val_size > 0:
        train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
        print(f"Dataset split: {train_size} items for training, {val_size} items for validation.")
    else:
        train_dataset = full_dataset
        val_dataset = None
        print(f"Dataset: {train_size} items for training (no validation).")
    
    # Use multiple workers for parallel data loading (CPU) and pin_memory for faster GPU transfer
    use_workers = min(args.num_workers, 8) if args.num_workers > 0 else 8
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                              num_workers=use_workers, pin_memory=True, prefetch_factor=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                            num_workers=use_workers, pin_memory=True) if val_dataset else None
    
    trainer = MatSAMTrainer(args)
    trainer.train(train_loader, val_loader)

if __name__ == "__main__":
    main()
