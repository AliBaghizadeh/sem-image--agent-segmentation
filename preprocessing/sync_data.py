
"""
Dataset Sync Tool: Ensures that the images folder and masks folder are perfectly matched.
If you manually deleted "bad" masks from your masks folder, run this script to 
automatically remove (or move) the corresponding images so the training set remains synced.

Typical usage:
    python preprocessing/sync_data.py --image_dir "data/tiled_images" --mask_dir "data/tiled_masks"
"""

import os
import argparse
from pathlib import Path
import shutil
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Sync image and mask folders by removing orphans.")
    parser.add_argument("--image_dir", type=str, required=True, help="Folder containing tiled images.")
    parser.add_argument("--mask_dir", type=str, required=True, help="Folder containing your cleaned/selected masks.")
    parser.add_argument("--action", type=str, choices=["move", "delete"], default="move", 
                        help="Action to take with orphan images. 'move' puts them in a backup folder, 'delete' removes them permanently. Default: move")
    return parser.parse_args()

def main():
    args = parse_args()
    img_dir = Path(args.image_dir)
    mask_dir = Path(args.mask_dir)
    
    if not img_dir.exists() or not mask_dir.exists():
        print(f"❌ Error: One or both directories do not exist.")
        return

    # 1. Gather all mask stems (filenames without extension)
    # We use stems because images might be .tif and masks might be .png
    mask_stems = {p.stem for p in mask_dir.glob("*") if p.is_file()}
    
    # 2. Gather all image files
    extensions = [".png", ".jpg", ".jpeg", ".tif", ".tiff"]
    img_files = []
    for ext in extensions:
        img_files.extend(list(img_dir.glob(f"*{ext}")))
    
    print(f"📊 Found {len(mask_stems)} masks and {len(img_files)} images.")

    # 3. Create backup folder if action is 'move'
    backup_dir = img_dir.parent / "orphan_images_backup"
    if args.action == "move":
        backup_dir.mkdir(parents=True, exist_ok=True)

    # 4. Filter and process
    orphans = []
    for img_p in img_files:
        if img_p.stem not in mask_stems:
            orphans.append(img_p)

    if not orphans:
        print("✅ Success: All images have corresponding masks. No cleanup needed.")
        return

    print(f"🧹 Found {len(orphans)} orphan images with no matching masks.")
    
    for img_p in tqdm(orphans, desc=f"{args.action.capitalize()}ing orphans"):
        try:
            if args.action == "move":
                shutil.move(str(img_p), str(backup_dir / img_p.name))
            else:
                img_p.unlink()
        except Exception as e:
            print(f"❌ Failed to {args.action} {img_p.name}: {e}")

    if args.action == "move":
        print(f"\n✅ Cleanup complete. {len(orphans)} images moved to: {backup_dir}")
    else:
        print(f"\n✅ Cleanup complete. {len(orphans)} images permanently deleted.")

if __name__ == "__main__":
    main()
