
import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.matsam.matsam_model import MatSAMModel
from core.measurements.grain_analysis import GrainAnalyzer

st.set_page_config(page_title="MicroSAM | Materials AI", layout="wide")

# Custom CSS for Premium Look
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stButton>button {
        background: linear-gradient(45deg, #00c6ff, #0072ff);
        color: white;
        border: none;
        padding: 10px 24px;
        border-radius: 8px;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 15px #00c6ff;
    }
    .card {
        background: rgba(255, 255, 255, 0.05);
        padding: 20px;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🔬 MicroSAM: Agentic Microstructure Intelligence")
st.markdown("---")

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🎯 Control Center")
    mode = st.radio("Select Mode", ["Interactive Analysis", "Autonomous Batch Trainer"])
    
    st.markdown("---")
    st.subheader("⚖️ Model Weights")
    weight_mode = st.toggle("Use Fine-tuned Weights", value=False)
    
    custom_weight_path = "finetuning/runs/exp1/best_model.pth"
    if weight_mode:
        if os.path.exists(custom_weight_path):
            st.success(f"Custom weights found!")
            weight_to_use = custom_weight_path
        else:
            st.error("best_model.pth not found in finetuning/runs/exp1/. Using base weights.")
            weight_to_use = "models/sam_weights/sam_vit_l_0b3195.pth"
    else:
        weight_to_use = "models/sam_weights/sam_vit_l_0b3195.pth"
    
    st.markdown("---")
    uploaded_file = st.file_uploader("Upload SEM Image", type=["png", "jpg", "jpeg", "tif"])
    
    user_query = st.text_input("Agent Query", placeholder="e.g., 'Analyze grains larger than 50um'")
    
    process_btn = st.button("🚀 Process Image")
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        st.image(img_rgb, caption="Input SEM Image", use_container_width=True)
        
        if process_btn:
            with st.spinner("MatSAM is analyzing microstructure..."):
                # Initialize Model based on selection
                try:
                    matsam = MatSAMModel(checkpoint_path=weight_to_use)
                    masks = matsam.generate_auto_masks(img_rgb)
                    
                    st.success(f"Analysis Complete! Found {len(masks)} grains.")
                    
                    # Layout for results
                    res_col1, res_col2 = st.columns(2)
                    
                    with res_col1:
                        st.subheader("Segmentation Overlay")
                        # Draw contours
                        overlay = img_rgb.copy()
                        for mask_data in masks:
                            m = mask_data['segmentation'].astype(np.uint8)
                            contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
                        
                        st.image(overlay, caption="Delineated Grains", use_container_width=True)
                    
                    with res_col2:
                        st.subheader("Morphology Data")
                        analyzer = GrainAnalyzer()
                        results_df = analyzer.analyze_masks(masks)
                        
                        st.dataframe(results_df, use_container_width=True)
                        st.download_button("Download Report (CSV)", data=results_df.to_csv(), file_name="analysis.csv")
                
                except Exception as e:
                    st.error(f"Error during analysis: {e}")

    else:
        st.info("Please upload an SEM image to begin.")
        
st.markdown("---")
st.caption("Developed for Advanced Materials Characterization | Powered by MatSAM & LangGraph")
