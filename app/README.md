# 🔬 SEM Agent Segmentation

SEM Agent is an advanced Scanning Electron Microscopy (SEM) image analysis tool. It combines the power of the **Segment Anything Model (SAM)** with a specialized **Agentic AI** layer to automate grain boundary detection and failure analysis.

---

## 🚀 Getting Started

### 1. Installation
Clone the repository and install the dependencies for both the core app and the AI Consultant:

```bash
git clone https://github.com/yourusername/MatSAM.git
cd SEM Agent/app
pip install -r requirements.txt
pip install -r requirements_ai.txt
```

### 2. Model Weights
By default, the app looks for MatSAM weights at:
`models/sam_weights/sam_vit_l_0b3195.pth`

If you don't have weights yet, you can launch the app and use the **📁 Browse Weights** button in the sidebar to select your local `.pth` file.

### 3. Launching the App
Run the following command from the `app/` directory:
```bash
streamlit run app.py
```

---

## 🌟 Key Features

### 🔍 Precision Segmentation
Detect grain boundaries in complex micrographs using our specialized MatSAM transformer, designed to handle the noise and low contrast typical of SEM imaging.

### 🧪 Enhancement Lab
Interactive "Laboratory" tab where you can fine-tune **Frangi Filter** and **Difference of Gaussians (DoG)** parameters. Use the "Side-by-Side Comparison" to see how enhancement improves the final mask.

### 🔬 AI Microstructure Consultant (RAG)
An integrated Expert Advisor that reads your research papers (PDFs). 
- **Ask anything**: "How should I optimize parameters for high-noise images?"
- **Context-Aware**: The AI analyzes your current slider positions and gives specific advice grounded in literature.

---

## 🏗️ Project Structure
- `app/app.py`: Main application UI and logic.
- `app/utils/ai_consultant.py`: Backend for the RAG-based AI Expert.
- `app/knowledge_base/`: Drop your research PDFs here for the AI to learn.
- `app/UI_CUSTOMIZATION_GUIDE.md`: Instructions for changing colors, fonts, and references.

---

## 📄 References & Citations
- **MATSAM Paper**: ["A Novel Training-Free Approach to Efficiently Extracting Material Microstructures Via Visual Large Model"](https://doi.org/10.1016/j.mattod.2023.xxx)
- **Methodology**: [Multi-scale Pre-processing for SEM Micrographs](https://medium.com/@alibaghizade/multi-scale-pre-processing-for-sem-micrographs-of-line-like-features-88303fb25631)

---

## 🤝 Contact & Contribution
- **Author**: Ali Baghi Zadeh
- 📧 [Email Me](mailto:alibaghizade@gmail.com)
- 🔗 [LinkedIn Profile](https://linkedin.com/in/yourprofile)
- 💻 [Project GitHub](https://github.com/yourusername)
