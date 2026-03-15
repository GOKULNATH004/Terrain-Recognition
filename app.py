"""
Streamlit Web UI for Terrain Recognition Model
Run with: streamlit run app.py
"""

import streamlit as st
import torch
import os
import sys
import tempfile
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add model directory to path
model_dir = os.path.join(os.path.dirname(__file__), 'model')
sys.path.insert(0, model_dir)

from inference import TerrainPredictor
from config import CHECKPOINT_DIR


# Page configuration
st.set_page_config(
    page_title="Terrain Recognition",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }
    .metric-card {
        background-color: #1f1f2e !important;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #4ECDC4;
    }
    .prediction-text {
        color: #FFD700 !important;
        font-size: 28px !important;
        font-weight: bold !important;
    }
    .confidence-text {
        color: #4ECDC4 !important;
        font-size: 24px !important;
        font-weight: bold !important;
    }
    .info-text {
        color: #E0E0E0 !important;
        font-size: 16px !important;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("🌍 Terrain Recognition AI")
st.markdown("""
### Classify terrain types using Deep Learning
Upload an image and our XCNN model will predict whether it's:
- 🏜️ **Desert** - Sandy, arid landscapes
- 🌲 **Forest** - Dense vegetation, trees
- ⛰️ **Mountain** - Rocky peaks, elevation
- 🌾 **Plains** - Flat grasslands
""")

# Sidebar
st.sidebar.title("Settings")
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold (%)", 
    min_value=0, 
    max_value=100, 
    value=50,
    help="Only show predictions above this confidence"
)

show_probabilities = st.sidebar.checkbox(
    "Show All Class Probabilities",
    value=True,
    help="Display confidence scores for all terrain types"
)

# Initialize model in cache for performance
@st.cache_resource
def load_model():
    """Load model once and cache it"""
    try:
        # Build path to best model
        model_path = os.path.join(
            os.path.dirname(__file__),
            'model',
            'checkpoints',
            'best_model.pth'
        )
        
        if not os.path.exists(model_path):
            return None, "Model not found. Please train the model first using: python model/train.py"
        
        predictor = TerrainPredictor(model_path=model_path)
        return predictor, None
    except Exception as e:
        return None, f"Error loading model: {str(e)}"


# Load model
with st.spinner("Loading model..."):
    predictor, error_msg = load_model()

if error_msg:
    st.error(f"❌ {error_msg}")
    st.info("📝 Steps to fix:")
    st.code("""
cd model
python train.py
    """)
    st.stop()

st.success("✅ Model loaded successfully!")

# Main interface
col1, col2 = st.columns(2)

with col1:
    st.subheader("📤 Upload Image")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png", "bmp"],
        help="Supports JPG, PNG, BMP formats. Recommended: 224x224+ pixels"
    )
    
    # Display uploaded image
    if uploaded_file:
        image = Image.open(uploaded_file)
        display_image = image.copy()
        display_image.thumbnail((300, 300))
        st.image(display_image, caption="Uploaded Image")

with col2:
    st.subheader("🎯 Prediction Results")
    
    if uploaded_file:
        # Make prediction
        with st.spinner("🔍 Analyzing image..."):
            try:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                    tmp_file.write(uploaded_file.getbuffer())
                    tmp_path = tmp_file.name
                
                # Make prediction using the temporary file path
                result = predictor.predict(tmp_path)
                
                # Clean up temporary file
                os.unlink(tmp_path)
                
                terrain_class = result['class']
                confidence = result['confidence']
                probabilities = result['probabilities']
                
                # Display main prediction
                if confidence >= confidence_threshold:
                    col_pred1, col_pred2 = st.columns(2)
                    
                    with col_pred1:
                        # Terrain emoji mapping
                        terrain_emoji = {
                            'Desert': '🏜️',
                            'Forest': '🌲',
                            'Mountain': '⛰️',
                            'Plains': '🌾'
                        }
                        emoji = terrain_emoji.get(terrain_class, '🌍')
                        
                        # Display prediction with styled box
                        st.markdown(f"""
                        <div class='metric-card'>
                            <div class='prediction-text'>{emoji} {terrain_class}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        st.caption("Predicted Terrain Type")
                    
                    with col_pred2:
                        # Confidence indicator
                        color = '#4ECB71' if confidence > 75 else '#FFD700' if confidence > 50 else '#FF6B6B'
                        st.markdown(f"""
                        <div class='metric-card'>
                            <div class='confidence-text'>{confidence:.1f}%</div>
                        </div>
                        """, unsafe_allow_html=True)
                        st.caption("Confidence Score")
                    
                    # Detailed analysis
                    st.markdown("---")
                    st.subheader("📊 Class Probabilities")
                    
                    if show_probabilities:
                        # Create probability visualization
                        classes = list(probabilities.keys())
                        probs = list(probabilities.values())
                        
                        # Create bar chart
                        fig, ax = plt.subplots(figsize=(10, 6))
                        colors = ['#FF6B6B', '#4ECDC4', '#95E1D3', '#FFE66D']
                        bars = ax.barh(classes, probs, color=colors)
                        
                        # Add value labels on bars
                        for i, (bar, prob) in enumerate(zip(bars, probs)):
                            ax.text(prob + 0.01, i, f'{prob*100:.1f}%', 
                                   va='center', fontweight='bold')
                        
                        ax.set_xlabel('Probability', fontsize=12, fontweight='bold')
                        ax.set_title('Terrain Classification Probabilities', 
                                    fontsize=14, fontweight='bold', pad=20)
                        ax.set_xlim(0, 1.1)
                        ax.grid(axis='x', alpha=0.3)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Probability table
                        st.markdown("#### Detailed Breakdown")
                        prob_data = {
                            'Terrain Type': classes,
                            'Probability': [f"{p*100:.2f}%" for p in probs],
                            'Confidence Level': [
                                '🟢 High (>60%)' if p > 0.6 else '🟡 Medium (30-60%)' if p > 0.3 else '🔴 Low (<30%)'
                                for p in probs
                            ]
                        }
                        st.dataframe(prob_data, width='stretch', hide_index=True)
                
                else:
                    st.warning(
                        f"⚠️ Confidence ({confidence:.1f}%) below threshold ({confidence_threshold}%)\n\n"
                        f"The model is uncertain. Prediction: **{terrain_class}**"
                    )
            
            except Exception as e:
                st.error(f"❌ Error during prediction: {str(e)}")
    else:
        st.info("👆 Upload an image to get started!")

# Information section
st.markdown("---")
st.subheader("ℹ️ About This Model")

col_info1, col_info2, col_info3 = st.columns(3)

with col_info1:
    st.markdown("""
    **Architecture**
    - XCNN with Residual Blocks
    - 20+ convolutional layers
    - Multi-scale feature learning
    """)

with col_info2:
    st.markdown("""
    **Training Data**
    - 4 terrain classes
    - Data augmentation applied
    - GPU optimized training
    """)

with col_info3:
    st.markdown("""
    **Device**
    - Runs on CUDA (GPU)
    - Fallback to CPU
    - Real-time predictions
    """)

# Footer
st.markdown("---")
col_footer1, col_footer2, col_footer3 = st.columns(3)

with col_footer1:
    st.markdown("🤖 **Model**: TerrainXCNN with Residual Blocks")

with col_footer2:
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
    st.markdown(f"💻 **Device**: {device_name}")

with col_footer3:
    st.markdown("⚡ **Framework**: PyTorch + Streamlit")
