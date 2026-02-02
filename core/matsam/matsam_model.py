
"""
Core MatSAM model wrapper for materials image segmentation.
Handles model loading, weight management (base/fine-tuned), and mask generation.

Example initialization:
    model = MatSAMModel(checkpoint_path="models/sam_weights/sam_vit_l_0b3195.pth")
"""

import os
import torch
import numpy as np
import cv2
from pathlib import Path
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

class MatSAMModel:
    def __init__(self, model_type="vit_l", checkpoint_path="models/sam_weights/sam_vit_l_0b3195.pth", device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path
        
        print(f"Initializing MatSAM with {model_type} on {self.device}...")
        self.sam = sam_model_registry[model_type](checkpoint=None) # Load structure first
        
        # Load weights
        if os.path.exists(checkpoint_path):
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            # Check if this is a full SAM model or just the fine-tuned bits
            # (Standard SAM has 'image_encoder.0.0.weight' etc, our fine-tune saves full sam.state_dict())
            try:
                self.sam.load_state_dict(state_dict)
                print(f"DONE: Loaded weights from {checkpoint_path}")
            except Exception as e:
                print(f"WARNING: Direct load failed ({str(e)[:200]}). Attempting partial load...")
                self.sam.load_state_dict(state_dict, strict=False)
        else:
            print(f"ERROR: Checkpoint not found at {checkpoint_path}")
            # Try loading the default if possible
            self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)

        self.sam.to(device=self.device)
        self.sam.eval() # CRITICAL: Ensure model is in evaluation mode
        
        self.predictor = SamPredictor(self.sam)
        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.sam,
            points_per_side=32,
            pred_iou_thresh=0.4,          # Lowered significantly (was 0.70)
            stability_score_thresh=0.7,   # More lenient (was 0.85)
            crop_n_layers=0,              # Disable tiling (images are already 1024x1024)
            min_mask_region_area=50,      # Smaller grains allowed
        )

    def generate_global_mask(self, image_rgb):
        """
        Generates a single global binary mask using the 'Null Prompt' logic 
        used during fine-tuning. This is much faster and bypasses grid-point issues.
        """
        # SAM image preprocessing
        from segment_anything.utils.transforms import ResizeLongestSide
        resizer = ResizeLongestSide(1024)
        input_image = resizer.apply_image(image_rgb)
        input_image_torch = torch.as_tensor(input_image, device=self.device).permute(2, 0, 1).unsqueeze(0).float()
        
        with torch.no_grad():
            # Apply same preprocessing as SAM
            input_image_torch = self.sam.preprocess(input_image_torch)
            
            # Encoder
            image_embedding = self.sam.image_encoder(input_image_torch)
            
            # Null Prompts
            sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
                points=None, boxes=None, masks=None
            )
            
            # Decoder
            low_res_masks, iou_predictions = self.sam.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=self.sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            
            # Upscale and threshold
            mask = self.sam.postprocess_masks(low_res_masks, (1024, 1024), image_rgb.shape[:2])[0, 0]
            binary_mask = (mask > 0).cpu().numpy().astype(np.uint8) * 255
            
        return binary_mask

    def generate_auto_masks(self, image, use_global=True):
        """
        Automatically segment grains. By default, uses the global pass for 
        fine-tuned models and the point-grid generator for the base model.
        """
        # Check if image is a path string, load it if so
        if isinstance(image, (str, Path)):
            img_p = str(image)
            image = cv2.imread(img_p)
            if image is None:
                raise ValueError(f"Could not load image from path: {img_p}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if use_global:
            
            global_mask = self.generate_global_mask(image)
            
            # --- Pre-processing for better separation ---
            # Morphological opening to break thin bridges
            kernel = np.ones((3,3), np.uint8)
            global_mask = cv2.morphologyEx(global_mask, cv2.MORPH_OPEN, kernel, iterations=1)
            
            # --- Watershed Post-processing to separate merged grains ---
            # 1. Distance Transform
            dist_transform = cv2.distanceTransform(global_mask, cv2.DIST_L2, 5)
            
            # 2. Threshold to find clear centers (Sure Foreground)
            # Lower threshold = more splitting (more seeds)
            ret, sure_fg = cv2.threshold(dist_transform, 0.05 * dist_transform.max(), 255, 0)
            sure_fg = np.uint8(sure_fg)
            
            # 3. Find unknown region (Sure Background - Sure Foreground)
            # For microstructure, the background is just 0
            unknown = cv2.subtract(global_mask, sure_fg)
            
            # 4. Marker labelling
            ret, markers = cv2.connectedComponents(sure_fg)
            markers = markers + 1 # Add one to all labels so that background is not 0, but 1
            markers[unknown == 255] = 0 # Now, mark the unknown region with 0
            
            # 5. Apply Watershed
            # Watershed needs BGR image
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            markers = cv2.watershed(image_bgr, markers)
            
            # Extract discrete masks from markers
            # Markers > 1 are the grains (0 is boundary, 1 is background)
            discrete_masks = []
            max_marker = markers.max()
            for m_idx in range(2, max_marker + 1):
                mask_i = (markers == m_idx).astype(np.uint8)
                area = np.sum(mask_i)
                if area < 50: continue
                
                # Get bbox
                coords = np.where(mask_i)
                if len(coords[0]) == 0: continue
                y1, y2, x1, x2 = coords[0].min(), coords[0].max(), coords[1].min(), coords[1].max()
                
                discrete_masks.append({
                    'segmentation': mask_i.astype(bool),
                    'area': float(area),
                    'bbox': [int(x1), int(y1), int(x2-x1), int(y2-y1)],
                    'predicted_iou': 1.0
                })
            
            return discrete_masks
        else:
            return self.mask_generator.generate(image)

    def predict_from_prompts(self, image, input_point=None, input_label=None, input_box=None):
        """
        Segment specific grains based on user prompts (clicks or boxes).
        """
        self.predictor.set_image(image)
        masks, scores, logits = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            box=input_box,
            multimask_output=True,
        )
        return masks, scores, logits
