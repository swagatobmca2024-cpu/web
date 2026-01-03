"""
Face Registration + Face Login with Liveness Detection (WebRTC)
Works on Streamlit Cloud
"""

import streamlit as st
import numpy as np
import cv2
import os
from deepface import DeepFace
from scipy.spatial.distance import cosine
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import threading

# ---------------- CONFIG ---------------- #
FACE_FILE = "admin_face.npy"
FACE_MATCH_THRESHOLD = 0.4
BLINK_EAR_THRESHOLD = 0.21
REQUIRED_BLINKS = 1
REQUIRED_HEAD_MOVES = 1
# ---------------------------------------- #

st.set_page_config(page_title="Face Login (WebRTC)", layout="centered")

# ---------------- SESSION STATE ---------------- #
for key, default in {
    "auth": False,
    "blink": 0,
    "head_move": 0,
    "prev_nose_x": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ---------------- MEDIAPIPE ---------------- #
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# ---------------- UTIL FUNCTIONS ---------------- #
def get_embedding(frame):
    try:
        rep = DeepFace.represent(
            img_path=frame,
            model_name="Facenet",
            enforce_detection=False
        )
        return np.array(rep[0]["embedding"])
    except Exception:
        return None

def eye_aspect_ratio(pts, idx):
    A = np.linalg.norm(pts[idx[1]] - pts[idx[5]])
    B = np.linalg.norm(pts[idx[2]] - pts[idx[4]])
    C = np.linalg.norm(pts[idx[0]] - pts[idx[3]])
    return (A + B) / (2.0 * C)

# ---------------- VIDEO PROCESSOR ---------------- #
class FaceProcessor(VideoTransformerBase):
    def __init__(self):
        self.last_frame = None
        self.lock = threading.Lock()

    def transform(self, frame: av.VideoFrame):
        img = frame.to_ndarray(format="bgr24")

        with self.lock:
            self.last_frame = img.copy()

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        if result.multi_face_landmarks:
            lm = result.multi_face_landmarks[0]
            pts = np.array([
                (int(p.x * img.shape[1]), int(p.y * img.shape[0]))
                for p in lm.landmark
            ])

            LEFT_EYE = [33, 160, 158, 133, 153, 144]
            RIGHT_EYE = [362, 385, 387, 263, 373, 380]

            ear = (
                eye_aspect_ratio(pts, LEFT_EYE) +
                eye_aspect_ratio(pts, RIGHT_EYE)
            ) / 2

            if ear < BLINK_EAR_THRESHOLD:
                st.session_state.blink += 1

            nose_x = pts[1][0]
            if st.session_state.prev_nose_x is not None:
                if abs(nose_x - st.session_state.prev_nose_x) > 15:
                    st.session_state.head_move += 1
            st.session_state.prev_nose_x = nose_x

            cv2.putText(img, f"Blinks: {st.session_state.blink}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(img, f"Head Moves: {st.session_state.head_move}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        return img

# ---------------- UI ---------------- #
st.title("🔐 Face Authentication (WebRTC)")

mode = st.radio("Select Mode", ["Register Face", "Face Login"])

# ================= REGISTER ================= #
if mode == "Register Face":
    st.subheader("📸 Register Admin Face")

    ctx = webrtc_streamer(
        key="register",
        video_transformer_factory=FaceProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_transform=True
    )

    if st.button("Save Face"):
        if ctx.video_transformer and ctx.video_transformer.last_frame is not None:
            emb = get_embedding(ctx.video_transformer.last_frame)
            if emb is not None:
                np.save(FACE_FILE, emb)
                st.success("✅ Face Registered Successfully")
            else:
                st.error("❌ Face not detected properly")
        else:
            st.error("❌ Camera not ready")

# ================= LOGIN ================= #
if mode == "Face Login":
    st.subheader("🎥 Live Face Login + Liveness")

    if not os.path.exists(FACE_FILE):
        st.warning("⚠️ No registered face found")
        st.stop()

    stored_embedding = np.load(FACE_FILE)

    ctx = webrtc_streamer(
        key="login",
        video_transformer_factory=FaceProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_transform=True
    )

    if (
        st.session_state.blink >= REQUIRED_BLINKS and
        st.session_state.head_move >= REQUIRED_HEAD_MOVES
    ):
        st.success("🟢 Liveness Check Passed")

        if st.button("Verify Face"):
            if ctx.video_transformer and ctx.video_transformer.last_frame is not None:
                emb = get_embedding(ctx.video_transformer.last_frame)
                if emb is not None:
                    dist = cosine(stored_embedding, emb)
                    if dist < FACE_MATCH_THRESHOLD:
                        st.session_state.auth = True
                        st.success("✅ Face Verified")
                    else:
                        st.error("❌ Face mismatch")
                else:
                    st.error("❌ Face not detected")
            else:
                st.error("❌ Camera not ready")

# ================= DASHBOARD ================= #
if st.session_state.auth:
    st.success("🎉 Hello Admin 👋")
    st.markdown("### Welcome to Admin Dashboard")
    st.markdown("""
✔️ Same Face  
✔️ Live Blink  
✔️ Head Movement
""")

    if st.button("Logout"):
        for k in ["auth", "blink", "head_move", "prev_nose_x"]:
            st.session_state[k] = False if k == "auth" else 0
