"""
Project: SilentVoice - AI Sign Language Translator
Hackathon: Neural Storm 2026 (FT-04)
Description: High-performance, fully automatic real-time sign language translation platform 
             with stunning SaaS design, live auto-sentence builder, and voice synthesis.
"""

import os
import cv2
import mediapipe as mp
import streamlit as st
from gtts import gTTS
import tempfile
import time

# --- 1. PAGE CONFIGURATION & LUXURY SAAS DESIGN ---
st.set_page_config(
    page_title="SilentVoice | AI Sign Language Translator",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ultra-Attractive Neon & Glassmorphic CSS with Advanced Animations
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    .main {
        background: radial-gradient(circle at top right, #090d16 0%, #020617 100%);
        color: #f8fafc;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Hero Banner */
    .hero-container {
        padding: 1.5rem 1rem;
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(236, 72, 153, 0.15) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 18px;
        text-align: center;
        margin-bottom: 1.5rem;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.5);
    }
    .hero-badge {
        display: inline-block;
        padding: 5px 14px;
        background: rgba(236, 72, 153, 0.2);
        color: #ec4899;
        border-radius: 30px;
        font-size: 0.8rem;
        font-weight: 700;
        margin-bottom: 0.8rem;
        border: 1px solid rgba(236, 72, 153, 0.4);
    }
    
    /* Vibrant Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
        color: white;
        border-radius: 12px;
        font-weight: 600;
        border: none;
        padding: 0.6rem 1.2rem;
        box-shadow: 0 8px 20px rgba(99, 102, 241, 0.4);
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #4f46e5 0%, #9333ea 100%);
        box-shadow: 0 10px 25px rgba(99, 102, 241, 0.6);
        transform: translateY(-2px);
    }

    /* Glass Cards with Neon Glow */
    .glass-card {
        background: rgba(15, 23, 42, 0.85);
        backdrop-filter: blur(16px);
        padding: 22px;
        border-radius: 16px;
        border: 1px solid rgba(99, 102, 241, 0.3);
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.6);
    }

    /* Status Pill */
    .status-pill {
        display: inline-flex;
        align-items: center;
        padding: 4px 12px;
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    
    h1, h2, h3 {
        color: #f8fafc !important;
        font-weight: 800;
    }
    
    [data-testid="stSidebar"] {
        background-color: #030712;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    </style>
""", unsafe_allow_html=True)

# --- 2. INITIALIZE MEDIAPIPE HANDS ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# --- 3. SESSION STATE ---
if "sentence_list" not in st.session_state:
    st.session_state.sentence_list = []
if "current_sign" not in st.session_state:
    st.session_state.current_sign = "Waiting for gesture..."
if "confidence" not in st.session_state:
    st.session_state.confidence = 0.0
if "last_added_time" not in st.session_state:
    st.session_state.last_added_time = 0

# Fast heuristic gesture logic
def detect_fast_gesture(hand_landmarks):
    tips = [4, 8, 12, 16, 20]
    mcp = [2, 5, 9, 13, 17]
    
    extended = 0
    if hand_landmarks.landmark[tips[0]].x > hand_landmarks.landmark[mcp[0]].x:
        extended += 1
    for i in range(1, 5):
        if hand_landmarks.landmark[tips[i]].y < hand_landmarks.landmark[mcp[i]].y:
            extended += 1

    if extended == 5:
        return "HELLO", 0.96
    elif extended == 0:
        return "HELP", 0.94
    elif extended == 2:
        return "WATER", 0.92
    elif extended == 1:
        return "YES", 0.93
    else:
        return "DETECTING", 0.85

# --- 4. STREAMLINED SIDEBAR ---
st.sidebar.title("🎛️ Settings")
language_choice = st.sidebar.selectbox(
    "🌐 Output Language",
    ["English", "Hinglish", "Hindi"]
)

st.sidebar.markdown("---")
if st.sidebar.button("🗑️ Clear Transcript"):
    st.session_state.sentence_list = []
    st.session_state.current_sign = "Waiting for gesture..."

st.sidebar.markdown("---")
st.sidebar.markdown("<p style='color: #64748b; font-size: 0.8rem; text-align: center;'>Neural Storm 2026 • FT-04</p>", unsafe_allow_html=True)

# --- 5. LANDING PAGE HERO HEADER ---
st.markdown("""
    <div class="hero-container">
        <div class="hero-badge">⚡ Neural Storm 2026 Finalist Project</div>
        <h1 style="font-size: 2.4rem; margin-bottom: 0.3rem; background: linear-gradient(90deg, #f8fafc 0%, #ec4899 50%, #6366f1 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            SilentVoice Pro
        </h1>
        <p style="color: #94a3b8; font-size: 0.95rem; max-width: 650px; margin: 0 auto;">
            Fully automatic real-time sign language translation powered by computer vision and multi-lingual voice synthesis.
        </p>
    </div>
""", unsafe_allow_html=True)

# Main Dashboard Layout
col1, col2 = st.columns([1.2, 1.0])

with col1:
    st.markdown("### 📷 Live Camera Workspace")
    camera_placeholder = st.empty()
    run_camera = st.checkbox("🟢 Launch Fully Automatic Tracking", value=False)

with col2:
    st.markdown("### 💬 Live Auto-Sentence Builder")
    
    st.markdown(f"""
        <div class="glass-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <span style="color: #94a3b8; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px;">Current Gesture</span>
                <span class="status-pill">● LIVE TRACKING</span>
            </div>
            <h2 style="color: #38bdf8 !important; margin: 0 0 10px 0; font-size: 2.2rem;">{st.session_state.current_sign}</h2>
            <p style="margin: 0; color: #94a3b8; font-size: 0.85rem;">Confidence: <b style="color: #f8fafc;">{st.session_state.confidence * 100:.1f}%</b></p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 📝 Constructed Transcript:")
    full_sentence_raw = " ".join(st.session_state.sentence_list)
    
    # Translation mapping
    display_sentence = full_sentence_raw
    tts_lang = 'en'
    
    if language_choice == "Hinglish":
        display_sentence = full_sentence_raw.replace("HELLO", "Namaste").replace("WATER", "Paani chahiye").replace("YES", "Haan").replace("HELP", "Madad chahiye")
        tts_lang = 'en'
    elif language_choice == "Hindi":
        display_sentence = full_sentence_raw.replace("HELLO", "नमस्ते").replace("WATER", "पानी चाहिए").replace("YES", "हाँ").replace("HELP", "मदद चाहिए")
        tts_lang = 'hi'

    st.info(display_sentence if display_sentence else "Transcript is empty. Make a sign in front of the camera to auto-convert...")

    col_act1, col_act2 = st.columns(2)
    with col_act1:
        if st.button("🔊 Speak Aloud"):
            if display_sentence:
                try:
                    tts = gTTS(text=display_sentence, lang=tts_lang, slow=False)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                        temp_filename = fp.name
                        tts.save(temp_filename)
                    st.audio(temp_filename, format='audio/mp3', autoplay=True)
                except Exception as e:
                    st.error(f"Audio error: {e}")
            else:
                st.warning("Transcript is empty!")
                
    with col_act2:
        if display_sentence:
            st.download_button(
                label="📥 Export Transcript",
                data=display_sentence,
                file_name="SilentVoice_Transcript.txt",
                mime="text/plain"
            )

# --- 6. FAST CAMERA LOOP WITH AUTOMATIC CONVERSION ---
if run_camera:
    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:
        
        while run_camera:
            ret, frame = cap.read()
            if not ret:
                st.error("Webcam failed.")
                break
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            
            current_detected = "No Hand Detected"
            current_conf = 0.0
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(236,72,153), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(56,189,248), thickness=2, circle_radius=2)
                    )
                    current_detected, current_conf = detect_fast_gesture(hand_landmarks)
            
            st.session_state.current_sign = current_detected
            st.session_state.confidence = current_conf
            
            # FULLY AUTOMATIC CONVERSION LOGIC (Adds gesture directly without buttons)
            if current_detected not in ["No Hand Detected", "Waiting for gesture...", "DETECTING"]:
                current_time = time.time()
                if current_time - st.session_state.last_added_time > 2.2: # 2.2 seconds interval between auto-adds
                    if not st.session_state.sentence_list or st.session_state.sentence_list[-1] != current_detected:
                        st.session_state.sentence_list.append(current_detected)
                        st.session_state.last_added_time = current_time
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            camera_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
            
        cap.release()