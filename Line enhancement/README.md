---
Title: SEM Contrast Enhancement
emoji: 📈
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: "1.36.0"
app_file: app/streamlit_app.py
pinned: false
---

# SEM Contrast Enhancement Settings

This repository condenses an SEM-preprocessing workflow into a single,
reproducible package plus a Streamlit demo. The goal is to showcase key image
enhancement algorithms (Frangi, DoG, CLAHE/bilateral, dirt inpainting) so large
SEM datasets can be cleaned before feeding them into downstream ML pipelines.

![Sample enhancement panels](docs/Image.png)

**Live demo:** https://huggingface.co/spaces/alibaghizade/SEM-contrast-enhancement               
**Deep dive:** [Medium – “Multi-scale Pre-processing for SEM Micrographs of Line-like Features”](https://medium.com/@alibaghizade/multi-scale-pre-processing-for-sem-micrographs-of-line-like-features-88303fb25631)

---
## 1. Architecture Overview

| Layer | Description |
| --- | --- |
| `sem_line_enhancer/loader.py` | Normalizes TIFF/PNG/JPG inputs to float32 `[0,1]`, removing metadata noise. |
| `sem_line_enhancer/enhancers.py` | Implements the four core algorithms: (1) multi-scale **Frangi** ridges, (2) **Difference-of-Gaussians** for mid-frequency contrast, (3) **CLAHE + bilateral** for grains/base texture, (4) **dirt blob detection & Telea inpainting** (regionprops + OpenCV). |
| `sem_line_enhancer/pipeline.py` | Orchestrates the Std-aware preprocessing pipeline and dual-path outputs (`lines`, `base`, `fused`). |
| `sem_line_enhancer/presets.py` | Stores experiment-backed hyperparameter sets for different SEM classes. |
| `sem_line_enhancer/cli.py` | CLI entrypoint for preprocessing (`.npy`) + PNG export, with optional MLflow logging. |
| `app/streamlit_app.py` | Upload/sample interface showing Original + Lines/Base/Fused panels and downloads. |

Two presets are shipped:
- **`lines`** – tuned for ferroelastic or needle-like features (Frangi+DoG-heavy, stronger dirt removal).
- **`boundaries`** – tuned for grain-only micrographs (lighter Frangi weights, CLAHE emphasis).

The app can use local PNG samples (`examples/`) or the built-in synthetic samples when running on Spaces.

---
## 2. Getting Started

```bash
# Install with dev extras (ruff/pytest already listed)
pip install -e .[dev] imagecodecs

# Quick CLI smoke test (line-focused preset)
python -m sem_line_enhancer.cli preprocess \
  --input examples \
  --output tmp \
  --limit 1 \
  --cpu-only \
  --preset lines

# Export the generated .npy files to PNG
python -m sem_line_enhancer.cli export \
  --input tmp \
  --output tmp_png \
  --types lines base fused

# Launch Streamlit app locally
streamlit run app/streamlit_app.py
```

The app lets users pick a preset, load one of the sample images (either the
bundled PNGs under `examples/` or the synthetic fallbacks shipped inside the app
for Spaces), or upload their own image. Results can be downloaded as `.npy`
arrays for downstream SAM/MatSAM workflows.

---
## 3. Experimentation & MLOps Hooks

### Grid Search
- Script: `gridsearch_single_preprocessing.py`
- Adjustable grids near the top (`FRANGI_*`, `DIRT_*`, `CLAHE_*`, `BILAT_*`). Two example experiments are documented in `docs/LOCAL_TESTING.md`:
  1. **Experiment 1** – broad Frangi/CLAHE sweep (lines + grains).
  2. **Experiment 2** – dirt-focused sweep (grain-only SEMs).
- Outputs land in `data/diagnostics/` (`gridsearch_stats.csv`, best-result panels, `best results.txt`).
- Promote winners by editing `sem_line_enhancer/presets.py`, then rerun the CLI/app to verify visually.

### MLflow (optional)
CLI flag `--mlflow` logs preset name, hyperparameters, and metadata to the
server defined in `MLFLOW_TRACKING_URI`. Example:
```bash
python -m sem_line_enhancer.cli preprocess \
  --input data/raw \
  --output data/preprocessed \
  --preset boundaries \
  --mlflow --mlflow-run-name boundary_sweep
```

### Continuous Integration / Deployment
- `.github/workflows/ci.yaml` installs `pip install -e .[dev] imagecodecs` and runs a CLI smoke test on every push/PR.
- `.github/workflows/cd.yaml` mirrors the repo to the Hugging Face Space whenever `main` changes.

---
## 4. Deployment Notes

### Local
- Use the CLI for batch preprocessing → `.npy` or PNG outputs.
- Run Streamlit locally (`streamlit run app/streamlit_app.py`) inside the Conda env (`ymno3_gpu`).

### GitHub
- Repo: `SEM-contrast-enhancement`. Push changes, CI runs automatically.
- `.gitignore` excludes raw data, diagnostics, and caches so only code/config/docs live in source control.

### Hugging Face Space (Streamlit)
1. Create a new HF Space (Streamlit template).
2. Connect it to this repo (or use the CD workflow above).
3. Optional secrets (MLflow tracking URI, AWS creds) can be stored via the HF Secrets tab—no tokens are committed.

Once deployed, the HF Space mirrors the local app: preset dropdown, sample images (real or synthetic), upload, download outputs.

---
## 5. Repository Layout

- `sem_line_enhancer/` – package modules (`cli.py`, `loader.py`, `pipeline.py`, `enhancers.py`, `presets.py`, …).  
- `app/` – Streamlit UI (`streamlit_app.py`) and app-specific docs.  
- `docs/LOCAL_TESTING.md` – CLI/grid-search command cookbook.  
- `gridsearch_single_preprocessing.py`, `resize_images.py` – optional experimentation utilities.  
- `examples/` – sample PNGs used locally (HF Space supplies its own copies).  
- `.github/workflows/` – CI (smoke test) + CD (deploy to HF Space).  
- `requirements.txt`, `pyproject.toml`, `README.md` – metadata and docs.

---

Feel free to fork or adapt to your own SEM datasets—the repo is intentionally modular and lightweight so students, PhDs, or hiring managers can inspect the complete workflow. Enjoy!
