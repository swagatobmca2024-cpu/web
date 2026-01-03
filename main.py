import streamlit as st
import numpy as np
import cv2
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

st.set_page_config(page_title="Face Login", layout="centered")

mp_face = mp.solutions.face_mesh

# -------------------------
# Utility functions
# -------------------------
def eye_aspect_ratio(landmarks, eye_ids):
    pts = np.array([[landmarks[i].x, landmarks[i].y] for i in eye_ids])
    vertical = np.linalg.norm(pts[1] - pts[5]) + np.linalg.norm(pts[2] - pts[4])
    horizontal = np.linalg.norm(pts[0] - pts[3])
    return vertical / (2.0 * horizontal)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# -------------------------
# Video Processor
# -------------------------
class FaceProcessor(VideoTransformerBase):
    def __init__(self):
        self.mesh = mp_face.FaceMesh(refine_landmarks=True)
        self.blinked = False
        self.embedding = None

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = self.mesh.process(rgb)

        if result.multi_face_landmarks:
            lm = result.multi_face_landmarks[0].landmark

            ear = (
                eye_aspect_ratio(lm, LEFT_EYE)
                + eye_aspect_ratio(lm, RIGHT_EYE)
            ) / 2

            if ear < 0.20:
                self.blinked = True

            self.embedding = np.array([[p.x, p.y, p.z] for p in lm]).flatten()

            h, w, _ = img.shape
            for p in lm:
                cv2.circle(img, (int(p.x * w), int(p.y * h)), 1, (0, 255, 0), -1)

        return img

# -------------------------
# Streamlit UI
# -------------------------
st.title("🛂 Face Registration & Login (Liveness)")

if "face_embed" not in st.session_state:
    st.session_state.face_embed = None

mode = st.radio("Select Mode", ["Register Face", "Login"])

ctx = webrtc_streamer(
    key="face",
    video_processor_factory=FaceProcessor,
    media_stream_constraints={"video": True, "audio": False},
)

if ctx.video_processor:
    vp = ctx.video_processor

    if st.button("Capture"):
        if vp.embedding is None:
            st.error("❌ No face detected")
        elif not vp.blinked:
            st.error("❌ Please blink for liveness")
        else:
            if mode == "Register Face":
                st.session_state.face_embed = vp.embedding
                st.success("✅ Face Registered Successfully")

            else:
                if st.session_state.face_embed is None:
                    st.error("❌ No registered face found")
                else:
                    dist = np.linalg.norm(
                        st.session_state.face_embed - vp.embedding
                    )
                    if dist < 6.0:
                        st.success("👋 Hello Admin – Welcome to Dashboard")
                    else:
                        st.error("❌ Face does not match")
