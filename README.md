# 🔬 SEM Agent Segmentation: AI-Powered Microstructure Analysis

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**SEM Agent Segmentation** (based on MatSAM) is an end-to-end framework for automated grain boundary detection in Scanning Electron Microscopy (SEM) images. It leverages a fine-tuned Segment Anything Model (SAM) with a specialized **Agentic AI** layer to automate grain boundary detection and failure analysis.

---

## 🌟 Key Features
- **🎯 Professional ML**: Fine-tuned SAM Mask Decoder specifically for materials science electron microscopy.
- **🛠️ Rescue Workflow**: Automated image enhancement for "failed" segmentations.
- **📊 Real-time Analytics**: Streamlit dashboard with grain size distribution and quality metrics.
- **🤖 Agentic Diagnostics**: AI agent that identifies segmentation failures and suggests fixes.
- **🔬 Ask from AI**: RAG-powered research advisor that interprets literature to suggest parameters.

---

## 📂 Project Structure
- `app/`: Interactive Streamlit application and AI agents.
- `core/`: Shared model wrappers and core logic.
- `finetuning/`: Training scripts and evaluation pipelines.
- `preprocessing/`: Data preparation and tiling utilities.
- `models/`: Model weight management (base SAM weights git-ignored).

---

## 🚀 Quick Start

### 1. Installation
```bash
git clone https://github.com/yourusername/sem-agent-segmentation.git
cd sem-agent-segmentation
pip install -r requirements.txt
```

### 2. Run the App
```bash
cd app
streamlit run app.py
```

### 3. Training/Data Prep
To retrain the model or process new data (e.g., 1024x921 tiles):
```bash
# Data Tiling
python preprocessing/tile_images.py --input_dir "data/raw" --output_dir "data/tiled" --tile_height 921 --tile_width 1024

# Fine-tuning
python finetuning/train_sam.py --train_data "data/tiled_images" --epochs 50
```

---

## 🧪 Methodology
This project implements a state-of-the-art microstructure analysis pipeline:
1. **SAM Adaptation**: Fine-tuning the mask decoder on specialized SEM grain datasets.
2. **Multi-scale Enhancement**: Utilizing Frangi and Difference-of-Gaussians (DoG) for line feature preservation (detailed in [Medium article](https://medium.com/@alibaghizade/multi-scale-pre-processing-for-sem-micrographs-of-line-like-features-88303fb25631)).
3. **Agentic Layer**: A diagnostic system that analyzes metadata and metrics to suggest "rescue" parameters.
4. **Expert Consultation**: A RAG-based LLM (**Ask from AI**) that acts as a domain specialist, grounded in published PDFs.

## 📄 Publications & References
- **Project Foundation**: ["A Novel Training-Free Approach to Efficiently Extracting Material Microstructures Via Visual Large Model"](https://doi.org/10.1016/j.mattod.2023.xxx) (MatSAM Paper)
- **Feature Engineering**: ["Multi-scale Pre-processing for SEM Micrographs of Line-like Features"](https://medium.com/@alibaghizade/multi-scale-pre-processing-for-sem-micrographs-of-line-like-features-88303fb25631) (Medium)

---

## 📜 Future Work
- Integration with LLM agents for natural language microstructure queries.
- Support for complex secondary phases and precipitates.

---

## 🤝 Contact & Contribution
- **Author**: Ali Baghi Zadeh
- 📧 [Email Me](mailto:alibaghizade@gmail.com)
- 🔗 [LinkedIn Profile](https://linkedin.com/in/baghizade)
- 💻 [Project GitHub](https://github.com/yourusername)

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
