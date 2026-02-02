import cv2
import os
from pathlib import Path
import numpy as np

import matplotlib.pyplot as plt

# ------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------
# Source SEM directory (set to the main raw folder in this repo)
INPUT_DIR = "data/raw"
# Destination for tiled crops (used later during preprocessing/app demos)
OUTPUT_DIR = "data/crops_512"
CROP_SIZE = 512                         # patch size
OVERLAP = 64                            # overlap to avoid cutting lines
PREVIEW = False                          # set False to disable preview selection
SAVE_ALL = True                      # if True, saves all patches without preview
# ------------------------------------------------------------------------


def create_output_folder():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


def load_image(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"[WARNING] Cannot read: {path}")
    return img


def crop_image(img, filename):
    h, w = img.shape
    crops = []
    crop_id = 0
    
    # Step along height
    for y in range(0, h - CROP_SIZE + 1, CROP_SIZE - OVERLAP):
        # Step along width
        for x in range(0, w - CROP_SIZE + 1, CROP_SIZE - OVERLAP):
            patch = img[y:y + CROP_SIZE, x:x + CROP_SIZE]
            crops.append((patch, f"{filename}_y{y}_x{x}.png"))
            crop_id += 1
    
    return crops



def preview_and_save(crops, base_out):
    """
    Shows each crop using matplotlib and asks user: Save this? (y/n)
    """
    selected = 0
    for patch, name in crops:
        # Show patch
        plt.figure(figsize=(4,4))
        plt.imshow(patch, cmap='gray')
        plt.title(f"Save this? y/n (q to quit)\n{name}")
        plt.axis('off')
        plt.tight_layout()
        plt.show(block=False)

        # Wait for user key input
        key = input("Save this crop? (y/n/q): ").strip().lower()

        # Close the figure
        plt.close()

        if key == 'y':
            cv2.imwrite(os.path.join(base_out, name), patch)
            selected += 1
        elif key == 'q':
            print("Exiting early...")
            break
        # n = skip

    print(f"[INFO] Saved {selected} selected crops.")



def save_all(crops, base_out):
    for patch, name in crops:
        cv2.imwrite(os.path.join(base_out, name), patch)
    print(f"[INFO] Saved all {len(crops)} crops.")


def main():
    create_output_folder()
    
    image_paths = list(Path(INPUT_DIR).glob("*.*"))
    print(f"[INFO] Found {len(image_paths)} images.")

    for idx, img_path in enumerate(image_paths):
        img = load_image(img_path)
        if img is None:
            continue

        stem = f"img_{idx:05d}"   # nice numeric ID
        crops = crop_image(img, stem)

        if PREVIEW and not SAVE_ALL:
            preview_and_save(crops, OUTPUT_DIR)
        else:
            save_all(crops, OUTPUT_DIR)


if __name__ == "__main__":
    main()
