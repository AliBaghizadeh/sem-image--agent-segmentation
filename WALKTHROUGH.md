# Walkthrough: Targeted Rescue Strategy

I have moved this guide into your workspace so you can modify it as we progress.

## Current Progress: Finding the "Rescue Preset"

We discovered that MatSAM often fails on dark, low-contrast SEM images, resulting in a "white blob" (a single mask covering the whole image). To fix this, we are using a **4-Model Enhancement Pipeline**:

1.  **Frangi**: Highlights the network of lines (grain boundaries).
2.  **DoG (Difference of Gaussians)**: Sharpens edges by subtracting blurred versions.
3.  **Dirt Filtering**: Removes small black artifacts that confuse SAM.
4.  **CLAHE**: Provides local contrast so SAM can "see" inside dark regions.

### Quantitative Success
Using the `gridsearch_single_image.py` script, we can generate dozens of variants. By running SAM on these, we proved that certain parameter sets can increase the grain count from **1** (failure) to **over 200** (success).

### Next Steps for You:
1.  Run the **Grid Search** on your target image.
2.  Scroll through the folder and find the **cleanest** boundary visualization.
3.  Use those numbers in **Step 4** of the `IMPLEMENTATION_PLAN.md` to confirm the final mask.
