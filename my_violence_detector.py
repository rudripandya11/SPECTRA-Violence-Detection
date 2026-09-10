import streamlit as st
import cv2
import numpy as np
import os
import time
from collections import deque
from tensorflow.keras.models import load_model
import tempfile

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="SPECTRA - Violence Detection System",
    page_icon="👁️",
    layout="wide"
)

# --- NEO-FUTURISTIC THEME AND STYLING ---
st.markdown("""
<style>
    /* Main Background */
    [data-testid="stAppViewContainer"], [data-testid="stSidebar"] {
        background-color: #0a0a1a;
    }

    /* High-contrast sidebar text */
    [data-testid="stSidebar"] span, [data-testid="stSidebar"] p {
        color: #FFFFFF !important;
        font-weight: bold;
    }

    /* Glassmorphism Container with Glowing Gradient Border */
    .custom-container {
        position: relative;
        padding: 25px;
        border-radius: 15px;
        margin-bottom: 25px;
        background: rgba(40, 40, 80, 0.3);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        z-index: 1;
        overflow: hidden;
    }

    /* The glowing border effect */
    .custom-container::before {
        content: '';
        position: absolute;
        top: 50%; left: 50%;
        width: 150%; height: 150%;
        background: conic-gradient(from 0deg, #8A2BE2, #4158D0, #00BFFF, #4158D0, #8A2BE2);
        z-index: -1;
        animation: rotate 5s linear infinite;
        transform: translate(-50%, -50%);
    }

    /* Glow animation */
    @keyframes rotate {
        100% {
            transform: translate(-50%, -50%) rotate(360deg);
        }
    }

    /* Main title text with gradient */
    .title-text {
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
        font-size: 3rem; /* Larger title */
        background: linear-gradient(45deg, #8A2BE2, #00BFFF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Header text with high contrast */
    .header-text {
        color: #FFFFFF;
        font-weight: bold;
        font-size: 1.6rem;
    }

    /* Body text with high contrast - CHANGED TO WHITE */
    .body-text, .body-text ol li {
        color: #FFFFFF;
        font-size: 1.1rem;
    }
    
    /* --- NEW: Big and Bold "Made By" Text --- */
    .made-by-text {
        text-align: center;
        font-size: 1.3rem;
        font-weight: bold;
        color: #FFFFFF;
        line-height: 1.8;
    }

    /* Larger, Bolder Alerts */
    [data-testid="stError"] {
        padding: 20px;
        font-size: 1.5rem;
        font-weight: bolder;
    }
    [data-testid="stWarning"] {
        padding: 15px;
        font-size: 1.2rem;
        font-weight: bold;
    }

    /* High-Contrast Metrics */
    [data-testid="stMetricLabel"] {
        color: #00aaff !important;
        font-weight: bold;
        font-size: 1.1rem;
    }
    [data-testid="stMetricValue"] {
        color: #FFFFFF !important;
        font-size: 2.75rem !important;
    }
    [data-testid="stInfo"] {
        background-color: transparent !important;
        color: #FFFFFF !important;
        font-size: 1.1rem !important;
    }
</style>
""", unsafe_allow_html=True)


# --- CONSTANTS ---
MODEL_PATH = "modelnew.h5"
CONFIDENCE_THRESHOLD = 0.95
CONSECUTIVE_FRAMES_THRESHOLD = 15


# --- MODEL LOADING ---
@st.cache_resource
def load_violence_model(model_path):
    try:
        model = load_model(model_path)
        return model
    except FileNotFoundError:
        st.error(f"Model file not found at '{model_path}'. Please ensure the model file is in the correct directory.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        return None

# --- UI PAGES ---
def render_home_page():
    st.markdown("<h1 class='title-text'>👁️ SPECTRA</h1>", unsafe_allow_html=True)
    
    with st.container():
        st.markdown("<div class='custom-container'>", unsafe_allow_html=True)
        st.markdown("<p class='header-text'>About SPECTRA</p>", unsafe_allow_html=True)
        # --- UPDATED HTML STRUCTURE AND TEXT ---
        st.markdown("""
        <div class='body-text'>
            <strong>Spectra</strong> is an advanced analytical tool that uses a sophisticated deep learning model to screen video footage and images for potential instances of violence. By leveraging cutting-edge computer vision, Spectra can help in identifying and flagging sensitive content, serving as a powerful preliminary analysis tool for content moderators, security analysts, and researchers.
            <br><br>
            <strong>How It Works</strong><br>
            The process is straightforward and designed for ease of use.
            <ol>
                <li><strong>Upload Your File:</strong> Click the "Browse files" button to upload a video or an image that you wish to analyze. The application supports common file formats like MP4, AVI, MOV, JPG, and PNG.</li>
                <li><strong>AI-Powered Analysis:</strong> Once uploaded, our machine learning model processes the file frame by frame. It analyzes patterns, movements, and objects within the content to predict the probability of violence.</li>
                <li><strong>Receive the Verdict:</strong> After the analysis is complete, Spectra will provide a clear classification, indicating whether the content is likely "Violent" or "Non-Violent."</li>
            </ol>
            <br>
            This tool demonstrates the potential of AI in content moderation and safety monitoring. Upload a file to begin your analysis.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with st.container():
        st.markdown("<div class='custom-container'>", unsafe_allow_html=True)
        st.markdown("<p class='header-text'>Developed By</p>", unsafe_allow_html=True)
        st.markdown("""
        <div class='made-by-text'>
        TANISHQ PATIL (IU2241230359)<br>
        RUDRI PANDYA (IU2241230358)<br>
        DHAIRYA CHAVDA (IU2241230350)
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

def render_detection_page(model):
    st.markdown("<h1 class='title-text'>SPECTRA Live Detection</h1>", unsafe_allow_html=True)
    
    alert_placeholder = st.empty()

    with st.container():
        st.markdown("<div class='custom-container'>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload a video to analyze...", type=["mp4", "avi", "mov", "mkv"])
        st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        video_path = tfile.name

        vs = None
        try:
            vs = cv2.VideoCapture(video_path)
            if not vs.isOpened():
                st.error("Error: Could not open video file.")
                return

            col1, col2 = st.columns([2, 1])
            with col1: video_frame_placeholder = st.empty()
            with col2: metrics_card_placeholder = st.empty()
            
            consecutive_violence_frames = 0
            violence_detected_permanently = False
            total_frames = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))
            current_frame_count = 0
            
            if total_frames > 0:
                progress_bar = st.progress(0, text="Analyzing video frames...")
            else:
                progress_bar = None

            while vs.isOpened():
                (grabbed, frame) = vs.read()
                if not grabbed: break

                current_frame_count += 1
                if progress_bar:
                    progress = min(current_frame_count / total_frames, 1.0)
                    progress_bar.progress(progress, text=f"Analyzing frame {current_frame_count}/{total_frames}...")

                output = frame.copy()
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, (128, 128)).astype("float32") / 255.0
                preds = model.predict(np.expand_dims(frame_resized, axis=0), verbose=0)[0]
                confidence = preds[0]
                
                if confidence > CONFIDENCE_THRESHOLD:
                    consecutive_violence_frames += 1
                else:
                    consecutive_violence_frames = 0
                
                if consecutive_violence_frames >= CONSECUTIVE_FRAMES_THRESHOLD:
                    violence_detected_permanently = True

                if violence_detected_permanently:
                    with alert_placeholder.container():
                        st.error("VIOLENCE ALERT: Take Immediate Action!")
                        st.warning("Sustained violent activity detected. Follow security protocols.")
                
                with col1:
                    if violence_detected_permanently:
                        cv2.rectangle(output, (0, 0), (550, 40), (0, 0, 0), -1)
                        cv2.putText(output, "** VIOLENCE DETECTED **", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                    video_frame_placeholder.image(output, channels="BGR", use_container_width=True)

                with col2:
                    with metrics_card_placeholder.container():
                        st.markdown("<div class='custom-container'>", unsafe_allow_html=True)
                        st.metric("💥 Violence Confidence", f"{confidence:.2%}")
                        st.info(f"**ℹ️ Consecutive Violent Frames:** {consecutive_violence_frames}")
                        st.markdown("</div>", unsafe_allow_html=True)
            
            if progress_bar:
                progress_bar.empty()
            
        finally:
            if vs is not None:
                vs.release()
            time.sleep(0.1)
            try:
                os.remove(video_path)
            except PermissionError:
                st.warning("⚠️ Could not delete temp file (still in use). It will be cleared later.")

# --- MAIN APP LOGIC ---
def main():
    model = load_violence_model(MODEL_PATH)
    if model is None:
        return

    with st.sidebar:
        st.title("👁️ SPECTRA")
        page = st.radio("Navigation", ["Home", "Detect Violence"], label_visibility="collapsed")

    if page == "Home":
        render_home_page()
    elif page == "Detect Violence":
        render_detection_page(model)

if __name__ == "__main__":
    main()
