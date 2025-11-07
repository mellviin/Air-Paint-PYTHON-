# server.py
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import cv2
import numpy as np
import mediapipe as mp
import threading
import time
from av import VideoFrame

st.set_page_config(page_title="AirDraw (Streamlit)", layout="wide")

st.title("AirDraw — Streamlit / OpenCV-style Demo")
st.write("Gesture-driven drawing. Uses MediaPipe (hands) and OpenCV. Note: camera runs in the browser via WebRTC.")

# Simple UI controls (left column)
with st.sidebar:
    st.header("Controls")
    brush_size = st.slider("Brush size", min_value=2, max_value=60, value=8)
    color_hex = st.color_picker("Brush color", "#ff3b30")
    tool = st.radio("Tool", ("Brush", "Eraser"))
    if st.button("Save current drawing"):
        st.session_state.get("save_request", 0)
        st.session_state.save_request = (st.session_state.get("save_request", 0) + 1)
    if st.button("Clear drawing"):
        st.session_state.clear_request = (st.session_state.get("clear_request", 0) + 1)
    st.write("---")
    st.write("Tip: index finger up to draw. Two fingers to select UI (client-side).")

# Global RTC config (optional TURN/STUN)
RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

# Transformer that receives frames and overlays the persistent draw_layer
class AirDrawTransformer(VideoTransformerBase):
    def __init__(self):
        # MediaPipe hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.6)
        self.mp_draw = mp.solutions.drawing_utils

        # Drawing state
        self.draw_layer = None
        self.lock = threading.Lock()
        self.xp, self.yp = None, None
        self.brush_size = 8
        self.color = (0, 59, 255)  # BGR for OpenCV default red-ish
        self.tool = "Brush"

        # Undo/redo
        self.history = []
        self.redo_stack = []
        self.HISTORY_LIMIT = 20

        # Track requests from Streamlit UI
        self._last_ui_state = {"save_counter": 0, "clear_counter": 0, "brush_size": 8, "color_hex": "#ff3b30", "tool": "Brush"}

    def _hex_to_bgr(self, hexcol):
        h = hexcol.lstrip('#')
        r, g, b = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
        return (b, g, r)

    def save_history(self):
        with self.lock:
            if self.draw_layer is not None:
                self.history.append(self.draw_layer.copy())
                if len(self.history) > self.HISTORY_LIMIT:
                    self.history.pop(0)
                self.redo_stack.clear()

    def undo(self):
        with self.lock:
            if len(self.history) > 1:
                self.redo_stack.append(self.history.pop())
                self.draw_layer[:] = self.history[-1].copy()

    def redo(self):
        with self.lock:
            if self.redo_stack:
                item = self.redo_stack.pop()
                self.history.append(item.copy())
                self.draw_layer[:] = item.copy()

    def clear(self):
        with self.lock:
            if self.draw_layer is not None:
                self.draw_layer[:] = 0
                self.save_history()

    def request_save_png(self):
        # returns a numpy BGR image (the drawing alone) to be saved by main thread
        with self.lock:
            if self.draw_layer is not None:
                return self.draw_layer.copy()
            else:
                return None

    def update_ui_state(self, ui_state):
        # Called from main thread to push UI changes
        self._last_ui_state = ui_state
        # apply brush size / color / tool changes
        self.brush_size = int(ui_state.get("brush_size", self.brush_size))
        self.color = self._hex_to_bgr(ui_state.get("color_hex", "#ff3b30"))
        self.tool = ui_state.get("tool", self.tool)

    def _fingers_up(self, hand_landmarks):
        tips = [4, 8, 12, 16, 20]
        fingers = []
        try:
            fingers.append(1 if hand_landmarks.landmark[tips[0]].x < hand_landmarks.landmark[tips[0]-1].x else 0)
        except:
            fingers.append(0)
        for i in range(1,5):
            try:
                fingers.append(1 if hand_landmarks.landmark[tips[i]].y < hand_landmarks.landmark[tips[i]-2].y else 0)
            except:
                fingers.append(0)
        return fingers

    def transform(self, frame: VideoFrame) -> VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)  # mirror to match webcam mirrors
        h, w, _ = img.shape

        # Initialize draw_layer once we know frame size
        with self.lock:
            if self.draw_layer is None:
                self.draw_layer = np.zeros_like(img)
                self.history = [self.draw_layer.copy()]

        # Update UI-controlled params if changed
        # (Main thread will call update_ui_state - but we also poll session_state every pass to be safe)
        try:
            ui = st.session_state
            ui_state = {
                "save_counter": ui.get("save_request", 0),
                "clear_counter": ui.get("clear_request", 0),
                "brush_size": ui.get("brush_size", self.brush_size),
                "color_hex": ui.get("color_hex", "#ff3b30"),
                "tool": ui.get("tool", "Brush"),
            }
            self.update_ui_state(ui_state)
        except Exception:
            pass

        # Hand tracking
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = self.hands.process(img_rgb)
        all_fingers_down = False

        if res.multi_hand_landmarks:
            for handLms in res.multi_hand_landmarks:
                # mp drawing for debug (optional)
                self.mp_draw.draw_landmarks(img, handLms, self.mp_hands.HAND_CONNECTIONS if hasattr(self, 'mp_hands') else None)

                index = handLms.landmark[8]
                mx, my = int(index.x * w), int(index.y * h)
                fingers = self._fingers_up(handLms)
                all_fingers_down = sum(fingers) == 0

                # Drawing with index finger up
                if fingers[1] == 1:
                    if self.xp is None or self.yp is None:
                        self.xp, self.yp = mx, my
                    # draw onto draw_layer
                    if self.tool == "Brush":
                        cv2.line(self.draw_layer, (self.xp, self.yp), (mx, my), self.color, int(self.brush_size))
                    else:  # Eraser
                        cv2.line(self.draw_layer, (self.xp, self.yp), (mx, my), (0,0,0), int(max(self.brush_size*3, 20)))
                    self.xp, self.yp = mx, my
                else:
                    # when user closes fist after drawing -> commit history
                    if self.xp is not None and self.yp is not None:
                        self.save_history()
                    self.xp, self.yp = None, None

        # Overlay draw_layer onto image
        with self.lock:
            overlay = cv2.addWeighted(img, 1.0, self.draw_layer, 1.0, 0)

        # Handle clear request (from main streamlit UI)
        try:
            ui = st.session_state
            if ui.get("clear_request", 0) != self._last_ui_state.get("clear_counter", 0):
                self.clear()
                self._last_ui_state["clear_counter"] = ui.get("clear_request", 0)
        except Exception:
            pass

        # Return frame to browser
        out_frame = VideoFrame.from_ndarray(overlay, format="bgr24")
        return out_frame

# Start webRTC streamer
webrtc_ctx = webrtc_streamer(
    key="airdraw",
    rtc_configuration=RTC_CONFIGURATION,
    video_transformer_factory=AirDrawTransformer,
    media_stream_constraints={"video": True, "audio": False},
    async_transform=True,
)

# Buttons that interact with transformer
if webrtc_ctx.state.playing:
    transformer = webrtc_ctx.video_transformer

    # Update transformer UI state from the sidebar controls
    ui_state = {
        "brush_size": brush_size,
        "color_hex": color_hex,
        "tool": tool,
        "save_counter": st.session_state.get("save_request", 0),
        "clear_counter": st.session_state.get("clear_request", 0),
    }
    if transformer:
        try:
            transformer.update_ui_state(ui_state)
        except Exception:
            pass

    # Undo / Redo / Save UI
    cols = st.columns(3)
    with cols[0]:
        if st.button("Undo"):
            if transformer:
                transformer.undo()
    with cols[1]:
        if st.button("Redo"):
            if transformer:
                transformer.redo()
    with cols[2]:
        if st.button("Export PNG"):
            if transformer:
                img_to_save = transformer.request_save_png()
                if img_to_save is not None:
                    ts = int(time.time())
                    fname = f"drawing_{ts}.png"
                    cv2.imwrite(fname, img_to_save)
                    st.success(f"Saved {fname}")
                    st.download_button("Download PNG", data=open(fname, "rb").read(), file_name=fname, mime="image/png")

else:
    st.info("Waiting for WebRTC connection... click 'Allow' to give camera access in the browser popup.")

st.write("If you run into build errors installing dependencies on Streamlit Cloud, see the notes in the README or fallback to the browser demo.")
