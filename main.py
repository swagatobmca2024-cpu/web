import streamlit as st
import numpy as np
import cv2
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

st.set_page_config(page_title="Face Login", layout="centered")

# ---------------- CONFIG ----------------
EAR_THRESHOLD = 0.20
FACE_DIST_THRESHOLD = 6.0

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

mp_face = mp.solutions.face_mesh

# ---------------- SESSION STATE ----------------
for k, v in {
    "registered_face": None,
    "blinked": False,
    "auth": False,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------------- UTIL ----------------
def eye_aspect_ratio(lm, idx):
    pts = np.array([[lm[i].x, lm[i].y] for i in idx])
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    return (A + B) / (2.0 * C)

# ---------------- VIDEO PROCESSOR ----------------
class FaceProcessor(VideoProcessorBase):
    def __init__(self):
        self.mesh = mp_face.FaceMesh(refine_landmarks=True)
        self.last_embedding = None
        self.blinked = False

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = self.mesh.process(rgb)

        if result.multi_face_landmarks:
            lm = result.multi_face_landmarks[0].landmark

            ear = (
                eye_aspect_ratio(lm, LEFT_EYE) +
                eye_aspect_ratio(lm, RIGHT_EYE)
            ) / 2

            if ear < EAR_THRESHOLD:
                self.blinked = True
                st.session_state.blinked = True

            self.last_embedding = np.array(
                [[p.x, p.y, p.z] for p in lm]
            ).flatten()

            h, w, _ = img.shape
            for p in lm:
                cv2.circle(
                    img,
                    (int(p.x * w), int(p.y * h)),
                    1,
                    (0, 255, 0),
                    -1
                )

        return img

# ---------------- UI ----------------
st.title("🔐 Face Registration & Login")

mode = st.radio("Mode", ["Register Face", "Login"])

ctx = webrtc_streamer(
    key="camera",
    video_processor_factory=FaceProcessor,
    media_stream_constraints={"video": True, "audio": False},
)

# ---------------- REGISTER ----------------
if mode == "Register Face":
    st.info("➡️ Look at camera and blink once")

    if st.button("📸 Register"):
        if not ctx.video_processor:
            st.error("Camera not ready")
        elif ctx.video_processor.last_embedding is None:
            st.error("No face detected")
        elif not st.session_state.blinked:
            st.error("Please blink for liveness")
        else:
            st.session_state.registered_face = ctx.video_processor.last_embedding
            st.session_state.blinked = False
            st.success("✅ Face Registered Successfully")

# ---------------- LOGIN ----------------
if mode == "Login":
    if st.session_state.registered_face is None:
        st.warning("⚠️ No face registered yet")
    else:
        st.info("➡️ Blink and click Login")

        if st.button("🔓 Login"):
            vp = ctx.video_processor
            if vp is None or vp.last_embedding is None:
                st.error("No face detected")
            elif not st.session_state.blinked:
                st.error("Blink required for liveness")
            else:
                dist = np.linalg.norm(
                    st.session_state.registered_face - vp.last_embedding
                )
                if dist < FACE_DIST_THRESHOLD:
                    st.session_state.auth = True
                    st.success("🎉 Login Successful")
                else:
                    st.error("❌ Face mismatch")

# ---------------- DASHBOARD ----------------
if st.session_state.auth:
    st.success("👋 Hello Admin")
    st.markdown("### Welcome to Admin Dashboard")
    st.markdown("✔️ Same Face\n✔️ Live Blink\n✔️ Secure Login")

    if st.button("🚪 Logout"):
        st.session_state.auth = False
