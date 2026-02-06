"""
SEM Microstructure Segmentation App with Agentic Diagnosis
Portfolio demonstration of ML-powered grain boundary detection with AI-driven failure analysis.

Author: Ali Baghi Zadeh
"""

import streamlit as st
import cv2
import numpy as np
import sys
from pathlib import Path
import torch
import io
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Import from your existing modules (using relative paths since we run from app/)
from utils.matsam_wrapper import MatSAM
from utils.visualization import create_overlay, plot_grain_distribution
from utils.metrics import calculate_quality_score
from agents.diagnostic_agent import SegmentationDiagnosticAgent
from utils.ai_consultant import get_consultant_instance

# Page config
st.set_page_config(
    page_title="SEM Agent Segmentation",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
<style>
    /* Global font size enforcement */
    html, body, [class*="st-"] {
        font-size: 16px !important;
    }

    /* Enforce maximum content width for consistency */
    [data-testid="stAppViewBlockContainer"] {
        max-width: 1920px !important;
        margin-left: auto !important;
        margin-right: auto !important;
    }
    
    .main-header {
        font-size: 3rem !important; /* Scaled up for 16px base */
        font-weight: 800;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }

    /* Ensure captions and small text are exactly 16px */
    .stCaption, .stMarkdown caption, [data-testid="stCaptionContainer"] {
        font-size: 1rem !important; 
        color: #4a5568 !important;
    }

    /* Metric and Sidebar scaling */
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 1rem !important;
    }
    [data-testid="stSidebar"] .stMarkdown p {
        font-size: 1rem !important;
    }

    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    /* UI Coherency: Standardized Height for single-line interactive elements */
    .stButton>button, 
    div[data-testid="stTextInput"] input {
        height: 3.2rem !important;
        line-height: 3.2rem !important;
        min-height: 3.2rem !important;
        max-height: 3.2rem !important;
        padding-top: 0 !important;
        padding-bottom: 0 !important;
    }

    /* Selectbox requires more flexible height to avoid squashing */
    div[data-testid="stSelectbox"] [data-baseweb="select"] {
        min-height: 3.2rem !important;
        border-radius: 8px !important;
    }

    /* Neutralize red borders on selection/focus with light blue */
    div[data-baseweb="select"] > div {
        border-color: rgba(0, 198, 255, 0.2) !important;
    }

    /* File Uploader requires more height to avoid text overlap */
    [data-testid="stFileUploader"] section {
        min-height: 5rem !important;
        padding: 1rem !important;
        border-radius: 8px !important;
    }

    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 400;
        font-size: 1rem !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
        width: auto !important;
        white-space: nowrap !important;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    /* Prevent sidebar buttons from overflowing */
    [data-testid="stSidebar"] [data-testid="stBaseButton-secondary"],
    [data-testid="stSidebar"] [data-testid="stBaseButton-primary"] {
        width: 100% !important;
        padding-left: 0.5rem !important;
        padding-right: 0.5rem !important;
    }

    /* Form field alignment */
    div[data-testid="stTextInput"] input,
    div[data-testid="stSelectbox"] [data-baseweb="select"] {
        border-radius: 8px !important;
    }

    [data-testid="stFileUploader"] section {
        border-radius: 8px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }

    /* Professional header scaling - Adjusted to not overshadow tabs */
    h1 { font-size: 2.5rem !important; }
    h2 { font-size: 2.0rem !important; }
    h3 { font-size: 1.5rem !important; }
    h4 { font-size: 1.1rem !important; }

    /* Replace Red Warning/Error colors with Light Blue */
    /* st.error override */
    [data-testid="stNotification"][data-status="error"] {
        background-color: #e1effe !important;
        color: #1e429f !important;
        border: 1px solid #a4cafe !important;
    }
    [data-testid="stNotification"][data-status="error"] svg {
        fill: #1e429f !important;
    }

    /* st.warning override */
    [data-testid="stNotification"][data-status="warning"] {
        background-color: #edf2f7 !important;
        color: #2b6cb0 !important;
        border: 1px solid #bee3f8 !important;
    }
    
    /* metric 'delta' red override */
    [data-testid="stMetricDelta"] > div [data-testid="stIcon"] {
        color: #00c6ff !important;
    }
    [data-testid="stMetricDelta"] > div {
        color: #00c6ff !important;
    }

    /* Custom 'Poor' status override for metrics if manually colored */
    .stMetric [data-testid="stMetricDelta"] svg {
        fill: #00c6ff !important;
    }

    /* Style Streamlit Tabs - Aggressive selector for all versions */
    [data-testid="stTabs"] button [data-testid="stMarkdownContainer"] p,
    [data-testid="stTabs"] button p,
    [data-testid="stTab"] p,
    [data-testid="stTab"] {
        font-size: 24px !important;
        font-weight: 800 !important;
        white-space: nowrap !important;
    }

    [data-testid="stTab"] {
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }
    
    /* Increase spacing between tab labels */
    [data-testid="stTabs"] {
        gap: 3rem !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'agent' not in st.session_state:
    st.session_state.agent = SegmentationDiagnosticAgent()
if 'results' not in st.session_state:
    st.session_state.results = None
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
if 'rescue_applied' not in st.session_state:
    st.session_state.rescue_applied = False
if 'lab_results' not in st.session_state:
    st.session_state.lab_results = None
if 'knowledge_indexed' not in st.session_state:
    st.session_state.knowledge_indexed = False

import tkinter as tk
from tkinter import filedialog

def load_model_ui():
    """UI component for selecting model weights, returns the loaded model."""
    st.markdown("#### 1️⃣ Model Configuration")
    st.caption("Default path to MatSAM weight in model/")
    
    # Define standard path
    base_dir = Path(__file__).resolve().parent.parent
    standard_weights = base_dir / "models/sam_weights/sam_vit_l_0b3195.pth"
    
    # Initialize session state path if not exists
    if 'current_model_path' not in st.session_state:
        st.session_state.current_model_path = str(standard_weights) if standard_weights.exists() else ""

    # Synchronize button logic
    if st.button("📁 Browse Weights..."):
        try:
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            selected_path = filedialog.askopenfilename(
                initialdir=str(standard_weights.parent if standard_weights.exists() else base_dir),
                title="Select MatSAM Weights (.pth)",
                filetypes=(("Model Weights", "*.pth"), ("All files", "*.*"))
            )
            root.destroy()
            
            if selected_path:
                # Update session state AND the specific key used for st.text_input to force UI sync
                st.session_state.current_model_path = selected_path
                st.session_state.path_input = selected_path  # This syncs with the text_input key
        except Exception as e:
            st.error(f"File picker error: {e}. Please enter path manually.")

    # Use a specific key 'path_input' to allow session state to control the value
    user_path = st.text_input(
        "Current Weights Path:",
        value=st.session_state.current_model_path,
        key="path_input",
        help="Full path to the .pth weights file"
    )
    
    # Ensure current_model_path stays in sync if user types
    st.session_state.current_model_path = user_path
    
    # Loading logic
    if st.session_state.current_model_path:
        checkpoint_path = Path(st.session_state.current_model_path)
        if checkpoint_path.exists():
            # Check if we need to load or it's a new path
            if st.session_state.model is None or st.session_state.get('loaded_path') != str(checkpoint_path):
                with st.spinner(f"Loading {checkpoint_path.name}..."):
                    try:
                        st.session_state.model = MatSAM(checkpoint=str(checkpoint_path))
                        st.session_state.loaded_path = str(checkpoint_path)
                        st.success("✅ Model Loaded Successfully")
                    except Exception as e:
                        st.error(f"❌ Error loading model: {e}")
        else:
            if st.session_state.current_model_path != "":
                st.warning("⚠️ File not found.")
    
    return st.session_state.model

def get_image_download_bytes(image_array, mode="RGB"):
    """Convert numpy array to proper PNG bytes for downloading."""
    if len(image_array.shape) == 2:
        # Grayscale
        pil_img = Image.fromarray(image_array.astype(np.uint8))
    else:
        # RGB
        pil_img = Image.fromarray(image_array.astype(np.uint8))
        
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()

# Import Line Enhancer (if available)
try:
    # Resolve absolute path to the 'Line enhancement' folder relative to this file
    # This makes the app portable as long as the repo structure is maintained
    repo_root = Path(__file__).resolve().parent.parent
    line_enhancement_dir = repo_root / "Line enhancement"
    
    if line_enhancement_dir.exists():
        if str(line_enhancement_dir) not in sys.path:
            sys.path.append(str(line_enhancement_dir))
            
        from sem_line_enhancer.pipeline import SEMPreprocessor
        from sem_line_enhancer.presets import PREPROCESSOR_PRESETS, PIPELINE_PRESETS
        HAS_ENHANCER = True
    else:
        print(f"DEBUG: Line enhancement directory not found at {line_enhancement_dir}")
        HAS_ENHANCER = False
except ImportError as e:
    print(f"DEBUG: Line enhancement import failed: {e}")
    # This usually happens if dependencies like 'scikit-image' or 'joblib' are missing
    HAS_ENHANCER = False
except Exception as e:
    print(f"DEBUG: Unexpected error loading enhancement module: {e}")
    HAS_ENHANCER = False

def process_image(image, model, enhance=True, use_global=False):
    """Run segmentation on uploaded image using exact CL pipeline."""
    # Convert PIL to numpy
    img_array = np.array(image)
    if len(img_array.shape) == 3:
        image_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        image_gray = img_array
    
    # 1. Enhancement (Matching command line test_one_image.py)
    if enhance and HAS_ENHANCER:
        try:
            preprocessor = SEMPreprocessor(**PREPROCESSOR_PRESETS["boundaries"])
            # Use defaults from command line script
            preset = PIPELINE_PRESETS["boundaries"].copy()
            preset["frangi_scales"] = [0.3, 0.7, 1.5]
            preset["clahe_clip"] = 10.0
            
            _, _, i_fused, _ = preprocessor.preprocess_dual(image_gray, **preset)
            
            # Use only enhanced for inference (blend=0.0 default in CL script)
            input_gray = (i_fused * 255).clip(0, 255).astype(np.uint8)
            input_rgb = cv2.cvtColor(input_gray, cv2.COLOR_GRAY2RGB)
            final_img_for_display = input_gray
        except Exception as e:
            st.error(f"Enhancement failed: {e}")
            input_rgb = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2RGB)
            final_img_for_display = image_gray
    else:
        input_rgb = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2RGB)
        final_img_for_display = image_gray

    # 2. Run Robust MatSAM (Matching command line script post-processing)
    try:
        mask = model.segment(input_rgb, use_global=use_global)
    except Exception as e:
        st.error(f"Segmentation error: {e}")
        mask = np.zeros_like(image_gray, dtype=np.uint8)
    
    # Analyze quality with agent
    agent = st.session_state.agent
    metrics = agent.analyze_quality(mask)
    
    return {
        'mask': mask,
        'metrics': metrics,
        'image_array': final_img_for_display,
        'raw_gray': image_gray
    }

def apply_rescue_workflow(image, parameters):
    """Apply rescue workflow with suggested parameters."""
    try:
        # Try to import your enhancement script
        from apply_enhancement import enhance_image
        enhanced = enhance_image(
            image,
            blend=parameters['blend'],
            clip=parameters['clip']
        )
    except:
        # Fallback: simple CLAHE enhancement
        clahe = cv2.createCLAHE(clipLimit=parameters['clip'], tileGridSize=(8,8))
        enhanced = clahe.apply(image)
    
    # Re-segment enhanced image
    if st.session_state.model:
        model = st.session_state.model
    else:
        st.error("Weights lost during rescue.")
        return None
        
    try:
        # Rescue ALWAYS uses Robust Grid mode for reliability
        mask = model.segment(enhanced, use_global=False)
    except:
        mask = np.zeros_like(enhanced, dtype=np.uint8)
    
    # Analyze new quality
    agent = st.session_state.agent
    metrics = agent.analyze_quality(mask)
    
    return {
        'mask': mask,
        'metrics': metrics,
        'image_array': enhanced,
        'enhanced': True
    }

# ==================== MAIN APP ====================

# Header
st.markdown('<h1 class="main-header">🔬 SEM Agent Segmentation</h1>', unsafe_allow_html=True)
st.markdown("**Automated grain boundary detection with intelligent failure diagnosis, based on MatSAM vision transformer**")

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/300x100/667eea/ffffff?text=SEM+Agent", use_container_width=True)
    
    st.markdown("### Settings")
    st.session_state.seg_mode = st.radio(
        "Segmentation Strategy",
        ["Robust (Point Grid)", "Fast (Experimental Global)"],
        index=0,
        help="Robust mode captures more detail but is slower. Global is instant but requires better fine-tuning."
    )
    
    st.session_state.do_enhance = st.checkbox(
        "Standard Enhancement",
        value=True,
        help="Apply Frangi + CLAHE pre-processing as in the command-line script."
    )
    
    st.markdown("---")
    st.markdown("### LLM settings")
    ai_provider = st.selectbox("LLM Provider", ["Ollama (Local)", "OpenAI", "Gemini"], index=0)
    
    if ai_provider == "Ollama (Local)":
        ollama_model = st.text_input("Ollama Model", value="llama3", help="Ensure this model is pulled in Ollama (e.g. 'ollama pull llama3')")
        ollama_url = st.text_input("Ollama Endpoint", value="http://localhost:11434")
        st.session_state.consultant_config = {"provider": "ollama", "model": ollama_model, "base_url": ollama_url}
    else:
        api_key = st.text_input("Enter API Key", type="password")
        st.session_state.consultant_config = {"provider": ai_provider.lower(), "api_key": api_key}
    
    # Initialize/Refresh Consultant
    consultant = get_consultant_instance(st.session_state)
    consultant.provider = st.session_state.consultant_config["provider"]
    if "model" in st.session_state.consultant_config:
        consultant.model = st.session_state.consultant_config["model"]
    if "base_url" in st.session_state.consultant_config:
        consultant.base_url = st.session_state.consultant_config["base_url"]
    
    if st.button("🔄 Reload Knowledge Base", use_container_width=True):
        with st.spinner("Indexing research papers..."):
            success, msg = consultant.initialize_rag()
            if success:
                st.success(msg)
            else:
                st.error(msg)

    st.markdown("---")
    st.markdown("### About This Project")
    st.info("""
    This app demonstrates:
    - 🤖 **Agentic AI**: Intelligent failure diagnosis
    - 🔬 **Domain Expertise**: SEM image analysis
    - 📊 **Data Visualization**: Interactive results
    - 🚀 **Production Code**: Clean, documented, deployable
    """)

# Main content
tab1, tab2, tab3 = st.tabs(["🔍 Segmentation + Agent", "🧪 Improve SEM Image Contrast", "🔬 Ask from AI"])

with tab1:
    # ------------------- INPUT SECTION (UNIFIED) -------------------
    st.markdown("### 🛠️ Configuration & Data")
    config_container = st.container(border=True)
    
    with config_container:
        conf_col1, conf_col2 = st.columns([1, 1])
        
        with conf_col1:
            model = load_model_ui()
            
        with conf_col2:
            st.markdown("#### 2️⃣ Image Upload")
            st.caption("Hint: For best results, use images tiled to 1024x1024 pixels.")
            uploaded_file = st.file_uploader(
                "Choose a SEM image",
                type=['tif', 'tiff', 'png', 'jpg', 'jpeg'],
                label_visibility="collapsed",
                help="Upload a scanning electron microscopy image for grain boundary detection"
            )
            
            if uploaded_file:
                image = Image.open(uploaded_file)
                st.session_state.uploaded_image = image
    
    # ------------------- ACTION SECTION -------------------
    # Logic to enforce workflow
    can_segment = st.session_state.model is not None and st.session_state.uploaded_image is not None
    
    if not st.session_state.model:
        st.warning("⚠️ **Workflow blocked:** Please select your model weights (.pth) above to enable segmentation.")
    elif not st.session_state.uploaded_image:
        st.info("ℹ️ Please upload an SEM image to begin.")
    
    if can_segment:
        if st.button("🚀 Run Segmentation", type="primary"):
            progress_bar = st.progress(0, text="Initializing SEM Agent...")
            
            with st.container():
                # 1. Preprocessing
                progress_bar.progress(10, text="🧪 Phase 1/3: Multi-scale Enhancement...")
                use_global = "Global" in st.session_state.seg_mode
                
                # We'll slightly refactor process_image logic into steps here for the progress bar
                img_array = np.array(st.session_state.uploaded_image)
                if len(img_array.shape) == 3:
                    image_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                else:
                    image_gray = img_array
                
                # Enhancement
                input_rgb = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2RGB)
                proc_img = image_gray
                if st.session_state.do_enhance and HAS_ENHANCER:
                    try:
                        preprocessor = SEMPreprocessor(**PREPROCESSOR_PRESETS["boundaries"])
                        preset = PIPELINE_PRESETS["boundaries"].copy()
                        preset["frangi_scales"] = [0.3, 0.7, 1.5]
                        preset["clahe_clip"] = 10.0
                        _, _, i_fused, _ = preprocessor.preprocess_dual(image_gray, **preset)
                        input_gray = (i_fused * 255).clip(0, 255).astype(np.uint8)
                        input_rgb = cv2.cvtColor(input_gray, cv2.COLOR_GRAY2RGB)
                        proc_img = input_gray
                    except: pass

                # 2. SAM Inference
                seg_msg = "🧠 Phase 2/3: SAM Vision Transformer (Robust Grid Mode)..." if not use_global else "🧠 Phase 2/3: SAM Vision Transformer (Global Pass)..."
                progress_bar.progress(30, text=seg_msg)
                
                try:
                    mask = st.session_state.model.segment(input_rgb, use_global=use_global)
                    progress_bar.progress(85, text="📊 Phase 3/3: Diagnostic Agent Analysis...")
                except Exception as e:
                    st.error(f"Segmentation error: {e}")
                    mask = np.zeros_like(image_gray, dtype=np.uint8)

                # 3. Post-processing & Agent
                agent = st.session_state.agent
                metrics = agent.analyze_quality(mask)
                
                results = {
                    'mask': mask,
                    'metrics': metrics,
                    'image_array': proc_img,
                    'raw_gray': image_gray
                }
                
                st.session_state.results = results
                st.session_state.rescue_applied = False
                
                progress_bar.progress(100, text="✅ Microstructure Analysis Complete!")
                st.success("✅ Segmentation complete!")
                # Remove progress bar after completion optionally, but here we keep it for a moment

    # ------------------- RESULTS DISPLAY -------------------
    if st.session_state.results:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(st.session_state.uploaded_image, caption="Original SEM Image", use_container_width=True)
            results = st.session_state.results
            metrics = results['metrics']
            
            st.markdown("### Results")
            
            # Dynamic Colors
            res_color = "#28a745" if metrics['is_good'] else "#dc3545"
            
            # Custom Results Metrics
            res_col1, res_col2, res_col3 = st.columns(3)
            with res_col1:
                st.markdown(f"""
                <div style="text-align: center;">
                    <p style="font-size: 18px; margin-bottom: 5px; color: #666;">Grain Count</p>
                    <p style="font-size: 32px; font-weight: 800; color: {res_color}; margin: 0;">{metrics['grain_count']}</p>
                </div>
                """, unsafe_allow_html=True)
            with res_col2:
                st.markdown(f"""
                <div style="text-align: center;">
                    <p style="font-size: 18px; margin-bottom: 5px; color: #666;">Coverage</p>
                    <p style="font-size: 32px; font-weight: 800; color: {res_color}; margin: 0;">{metrics['coverage']:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
            with res_col3:
                status_text = "Good" if metrics['is_good'] else "Poor"
                st.markdown(f"""
                <div style="text-align: center;">
                    <p style="font-size: 18px; margin-bottom: 5px; color: #666;">Quality Score</p>
                    <p style="font-size: 32px; font-weight: 800; color: {res_color}; margin: 0;">{metrics['quality_score']:.2f}</p>
                    <p style="font-size: 14px; color: {res_color}; margin-top: -5px; font-weight: 600;">{status_text}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Visualization
            st.markdown("### Segmentation Mask")
            overlay = create_overlay(results['image_array'], results['mask'])
            st.image(overlay, caption="Grain Boundaries Detected", use_container_width=True)
            
            # Download section
            st.markdown("#### 📥 Export Results")
            dl_col1, dl_col2 = st.columns(2)
            
            with dl_col1:
                mask_bytes = get_image_download_bytes(results['mask'])
                st.download_button(
                    label="📥 Binary Mask",
                    data=mask_bytes,
                    file_name=f"SEM_Agent_mask.png",
                    mime="image/png"
                )
                
            with dl_col2:
                overlay_bytes = get_image_download_bytes(overlay)
                st.download_button(
                    label="🎨 Visual Overlay",
                    data=overlay_bytes,
                    file_name=f"SEM_Agent_overlay.png",
                    mime="image/png"
                )
            
            # ========== AGENTIC DIAGNOSIS ==========
            st.markdown("---")
            st.markdown("### 🤖 AI Diagnostic Agent")
            
            # Generate diagnostic report
            agent = st.session_state.agent
            
            if not metrics['is_good'] and not st.session_state.rescue_applied:
                # Failure detected - show diagnosis and suggestions
                failure_type, diagnosis = agent.diagnose_failure(metrics)
                suggestion = agent.suggest_rescue_parameters(failure_type)
                
                # Show diagnosis
                st.warning(diagnosis)
                
                # Show agent's analysis
                with st.expander("🔍 Agent Analysis", expanded=True):
                    st.markdown(f"""
**Current Quality Metrics:**
- Grain Count: {metrics['grain_count']}
- Coverage: {metrics['coverage']:.1f}%
- Boundary Smoothness: {'Smooth' if metrics['jaggedness'] < 30 else 'Jagged'} (score: {metrics['jaggedness']:.1f})
- Overall Quality: {metrics['quality_score']:.2f}/1.00

**Diagnosed Issue:** {failure_type.replace('_', ' ').title()}
                    """)
                
                # Show suggested parameters
                with st.expander("🔧 Recommended Rescue Parameters", expanded=True):
                    st.markdown(suggestion['explanation'])
                    
                    # Show parameters in a nice format
                    param_col1, param_col2, param_col3 = st.columns(3)
                    with param_col1:
                        st.metric("Blend (B)", f"{suggestion['parameters']['blend']}")
                        st.metric("CLAHE Clip", f"{suggestion['parameters']['clip']}")
                    with param_col2:
                        st.metric("DoG σ_small", f"{suggestion['parameters']['sigma_small']}")
                        st.metric("DoG σ_large", f"{suggestion['parameters']['sigma_large']}")
                    with param_col3:
                        st.metric("Frangi Scale", f"{suggestion['parameters']['scale']}")
                
                # Apply rescue button
                if st.button("🔧 Apply Rescue Workflow", use_container_width=True, type="primary"):
                    with st.spinner("Applying rescue workflow..."):
                        rescued_results = apply_rescue_workflow(
                            results['image_array'],
                            suggestion['parameters']
                        )
                        st.session_state.results = rescued_results
                        st.session_state.rescue_applied = True
                        st.rerun()
            
            elif st.session_state.rescue_applied:
                # Rescue was applied - show improvement
                st.success("✅ **Rescue Workflow Applied Successfully!**")
                st.info(f"""
**Improved Quality Metrics:**
- Grain Count: {metrics['grain_count']}
- Coverage: {metrics['coverage']:.1f}%
- Quality Score: {metrics['quality_score']:.2f}/1.00
- Status: {'✅ Good' if metrics['is_good'] else '⚠️ Still needs attention'}
                """)
            
            else:
                # Quality is good
                st.success("✅ **Segmentation Quality: GOOD**")
                st.info(f"""
The segmentation looks excellent! No rescue workflow needed.

**Quality Metrics:**
- Grain Count: {metrics['grain_count']}
- Coverage: {metrics['coverage']:.1f}%
- Quality Score: {metrics['quality_score']:.2f}/1.00
                """)

with tab2:
    st.markdown("### 🧪 Improve SEM Image Contrast")
    st.markdown("Fine-tune the advanced preprocessing parameters used during training to optimize your microstructure visibility. You can use GridSearch approach provided in GitHub repo for wide ranges of parameters. Use Ask from AI tab to ask about parameters using your choice of LLM model")
    
    if not HAS_ENHANCER:
        st.error("Line enhancement module not found. Please ensure the 'Line enhancement' folder is in the project root.")
    else:
        enhance_col1, enhance_col2 = st.columns([1, 2])
        
        with enhance_col1:
            st.markdown("#### ⚙️ Core Parameters to Tune")
            lab_blend = st.slider("Global Blending", 0.0, 1.5, 0.0, 0.1, help="0.0 = Pure Enhanced, 1.0 = Pure Raw, >1.0 = Boosted Raw Contrast")
            lab_clip = st.slider("CLAHE Clip Limit", 1.0, 30.0, 15.0, 0.5)
            
            # Error handling for scales
            lab_scales = st.multiselect(
                "Frangi Scales", 
                [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0], 
                default=[0.3, 0.7, 1.5],
                help="Select scales to detect boundaries of different widths."
            )
            
            if not lab_scales:
                st.warning("⚠️ **Frangi disabled**: Please select at least one scale to use Frangi filters.")
            
            st.markdown("#### 🛠️ Advanced R&D Settings")
            with st.expander("Fine-Tuning Parameters"):
                lab_dog_small = st.slider("DoG Sigma Small", 0.1, 5.0, 1.0, 0.1)
                lab_dog_large = st.slider("DoG Sigma Large", 1.0, 25.0, 4.0, 0.5)
                lab_dirt_thresh = st.slider("Dirt Threshold", 0.01, 1.0, 0.1, 0.05, help="Lower = more aggressive cleaning of artifacts.")
        
        with enhance_col2:
            lab_image_file = st.file_uploader("Upload image for parameter testing", type=['tif', 'png', 'jpg'], key="lab_uploader")
            
            if lab_image_file:
                lab_img = Image.open(lab_image_file)
                lab_array = np.array(lab_img)
                if len(lab_array.shape) == 3:
                    lab_gray = cv2.cvtColor(lab_array, cv2.COLOR_RGB2GRAY)
                else:
                    lab_gray = lab_array
                
                with st.spinner("Calculating optimized view..."):
                    try:
                        lab_preprocessor = SEMPreprocessor(**PREPROCESSOR_PRESETS["boundaries"])
                        # Build dynamic preset
                        lab_preset = PIPELINE_PRESETS["boundaries"].copy()
                        lab_preset["use_frangi"] = len(lab_scales) > 0
                        lab_preset["frangi_scales"] = lab_scales if lab_scales else [1.0] # Fallback to prevent crash
                        lab_preset["w_frangi"] = 1.0 # Guarantee visibility in Lab
                        lab_preset["clahe_clip"] = lab_clip
                        lab_preset["use_dog"] = True
                        lab_preset["w_dog"] = 1.0   # Guarantee visibility in Lab
                        lab_preset["dog_sigma_small"] = lab_dog_small
                        lab_preset["dog_sigma_large"] = lab_dog_large
                        lab_preset["dirt_threshold"] = lab_dirt_thresh
                        
                        # Generate variants
                        i_lines, i_base, lab_fused, _ = lab_preprocessor.preprocess_dual(lab_gray, **lab_preset)
                        
                        # Final blending logic for lab
                        lab_fused_vis = (lab_fused * 255).clip(0, 255).astype(np.uint8)
                        if lab_blend > 0:
                            lab_final = cv2.addWeighted(lab_gray, lab_blend, lab_fused_vis, 1-lab_blend, 0)
                        else:
                            lab_final = lab_fused_vis
                            
                        st.markdown("#### 🔬 Benchmarking")
                        # We'll use a 2x2 grid approach by creating 2 rows of 2 columns
                        bench_row1_col1, bench_row1_col2 = st.columns(2)
                        bench_row2_col1, bench_row2_col2 = st.columns(2)
                        
                        with bench_row1_col1:
                            st.image(lab_gray, caption="1. Original Raw", use_container_width=True)
                        with bench_row1_col2:
                            st.image(lab_final, caption="2. Enhanced Result", use_container_width=True)
                        
                        # --- LAB SEGMENTATION WORKFLOW ---
                        with enhance_col1:
                            st.markdown("#### 🚀 Verify Quality")
                            st.info("Generating comparison masks is CPU/GPU intensive. Click below to benchmark your current sliders.")
                            
                            if st.button("🔬 Run Side-by-Side Comparison", type="secondary"):
                                if st.session_state.model is None:
                                    st.warning("⚠️ Load weights in Tab 1 first.")
                                else:
                                    lab_progress = st.progress(0, text="Starting comparison analysis...")
                                    
                                    # 1. Mask from Raw
                                    lab_progress.progress(10, text="🧠 Analyzing Raw image (1/2)...")
                                    raw_rgb = cv2.cvtColor(lab_gray, cv2.COLOR_GRAY2RGB)
                                    mask_raw = st.session_state.model.segment(raw_rgb)
                                    overlay_raw = create_overlay(lab_gray, mask_raw)
                                    
                                    # 2. Mask from Enhanced
                                    lab_progress.progress(50, text="🧠 Analyzing Enhanced image (2/2)...")
                                    lab_input_rgb = cv2.cvtColor(lab_final, cv2.COLOR_GRAY2RGB)
                                    mask_enh = st.session_state.model.segment(lab_input_rgb)
                                    overlay_enh = create_overlay(lab_final, mask_enh)
                                    
                                    # Store in session state to persist through UI reruns
                                    st.session_state.lab_results = {
                                        "overlay_raw": overlay_raw,
                                        "overlay_enh": overlay_enh,
                                        "params_used": lab_preset.copy()
                                    }
                                    lab_progress.progress(100, text="✅ Comparison analysis complete!")

                        # Display results if they exist
                        if st.session_state.lab_results is not None:
                            with bench_row2_col1:
                                st.image(st.session_state.lab_results["overlay_raw"], caption="3. Mask (From Raw)", use_container_width=True)
                            with bench_row2_col2:
                                st.image(st.session_state.lab_results["overlay_enh"], caption="4. Mask (From Enhanced)", use_container_width=True)
                            
                            st.success("✨ Comparison complete! If you change sliders, click the button again to update.")
                            
                            if st.button("🗑️ Clear Lab Results"):
                                st.session_state.lab_results = None
                                st.rerun()
                        else:
                            with bench_row2_col1:
                                st.info("Step 3: Raw impact")
                            with bench_row2_col2:
                                st.info("Step 4: Enhanced impact")

                        # Download Section
                        st.markdown("---")
                        dl_lab1, dl_lab2 = st.columns(2)
                        with dl_lab1:
                            lab_final_bytes = get_image_download_bytes(lab_final)
                            st.download_button(
                                "📥 Download Enhanced PNG",
                                data=lab_final_bytes,
                                file_name="enhanced_experiment.png",
                                mime="image/png"
                            )
                        with dl_lab2:
                            st.info("💡 Tip: Use these parameters to fix images where the base model fails.")
                            
                    except Exception as e:
                        st.error(f"Processing error: {e}. Check your parameter ranges.")
            else:
                st.info("💡 Please upload an image for parameter testing above to start fine-tuning.")

with tab3:
    st.markdown("### 🔬 Ask from AI")
    st.markdown("""
    Ask our AI Expert for advice on preprocessing your SEM images. 
    The consultant uses **RAG (Retrieval-Augmented Generation)** to ground its answers in your research papers.
    """)
    
    consultant = get_consultant_instance(st.session_state)
    
    if not consultant.is_initialized:
        st.info("💡 **Tip**: Click 'Reload Knowledge Base' in the sidebar to index your research papers for better advice.")
    
    # Chat Interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("How should I optimize Frangi scales for high-noise images?"):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Prepare context (current metrics/sliders)
        context = {
            "current_sliders": {
                "frangi_scales": st.session_state.get('lab_scales', 'Not set'),
                "dog_sigma_small": st.session_state.get('lab_dog_small', 'Not set'),
                "dog_sigma_large": st.session_state.get('lab_dog_large', 'Not set'),
                "blend": st.session_state.get('lab_blend', 'Not set')
            }
        }
        if st.session_state.get('results'):
            context["metrics"] = st.session_state.results['metrics']

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Consulting research papers..."):
                response = consultant.query(prompt, context=context)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

# Refactor "About" to an Expander at the bottom
st.markdown("---")
with st.expander("📖 About SEM Agent & Technical Methodology"):
    st.markdown("""
    ## Problem Statement
    Manual grain boundary annotation in SEM images is time-consuming and subjective. 
    This project automates the process using a fine-tuned Segment Anything Model (SAM)
    with an AI agent that diagnoses failures and suggests optimal preprocessing.
    
    ## Methodology
    1. **Data Preparation**: Tiled large SEM images into 1024x921 patches
    2. **Baseline**: Applied pre-trained SAM (failed on ~30% of images)
    3. **Rescue Workflow**: Developed targeted preprocessing for failed cases
    4. **Fine-Tuning**: Trained SAM decoder on rescued masks
    5. **Agentic Layer**: Built diagnostic agent for automatic failure analysis
    6. **Deployment**: Created this production-ready application
    
    ## Agentic Features
    The diagnostic agent:
    - Analyzes segmentation quality in real-time
    - Diagnoses failure mode
    - Suggests optimal rescue parameters based on failure type
    - Automatically applies enhancement workflow
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Built with ❤️ using Streamlit + AI Agents | © 2026 Ali [Your Name]</p>
</div>
""", unsafe_allow_html=True)
