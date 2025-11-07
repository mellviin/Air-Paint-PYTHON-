import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import cv2
import numpy as np
import mediapipe as mp
import threading
import time
from av import VideoFrame

st.set_page_config(page_title="AirDraw (Render)", layout="wide")

st.title("AirDraw — Render / Streamlit WebRTC")
st.write("Gesture-driven drawing. Uses MediaPipe (hands) and OpenCV. Camera runs in the browser via WebRTC.")

with st.sidebar:
    st.header("Controls")
    brush_size = st.slider("Brush size", min_value=2, max_value=60, value=8)
    color_hex = st.color_picker("Brush color", "#ff3b30")
    tool = st.radio("Tool", ("Brush", "Eraser"))
    if st.button("Clear drawing"):
        st.session_state.clear_request = st.session_state.get("clear_request", 0) + 1
    st.write("---\nTip: index finger up to draw. Two fingers to select UI (client-side).")

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

class AirDrawTransformer(VideoTransformerBase):
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.6)
        self.mp_draw = mp.solutions.drawing_utils

        self.draw_layer = None
        self.lock = threading.Lock()
        self.xp, self.yp = None, None
        self.brush_size = 8
        self.color = (0, 59, 255)
        self.tool = "Brush"

        self.history = []
        self.redo_stack = []
        self.HISTORY_LIMIT = 30

        self._last_clear = 0

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
        with self.lock:
            if self.draw_layer is not None:
                return self.draw_layer.copy()
            return None

    def update_ui(self, brush_size=None, color_hex=None, tool=None, clear_counter=None):
        if brush_size is not None:
            self.brush_size = int(brush_size)
        if color_hex is not None:
            self.color = self._hex_to_bgr(color_hex)
        if tool is not None:
            self.tool = tool
        if clear_counter is not None and clear_counter != self._last_clear:
            self.clear()
            self._last_clear = clear_counter

    def _fingers_up(self, hand_landmarks):
        tips = [4,8,12,16,20]
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
        img = cv2.flip(img, 1)
        h, w, _ = img.shape

        with self.lock:
            if self.draw_layer is None:
                self.draw_layer = np.zeros_like(img)
                self.history = [self.draw_layer.copy()]

        # Read UI state from session_state (main thread)
        try:
            ui = st.session_state
            self.update_ui(brush_size=ui.get("brush_size", self.brush_size),
                           color_hex=ui.get("color_hex", "#ff3b30"),
                           tool=ui.get("tool", self.tool),
                           clear_counter=ui.get("clear_request", self._last_clear))
        except Exception:
            pass

        # Hand tracking
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = self.hands.process(img_rgb)

        if res.multi_hand_landmarks:
            for handLms in res.multi_hand_landmarks:
                # optional drawing of landmarks for debugging:
                # self.mp_draw.draw_landmarks(img, handLms, self.mp_hands.HAND_CONNECTIONS)
                index = handLms.landmark[8]
                mx, my = int(index.x * w), int(index.y * h)
                fingers = self._fingers_up(handLms)

                if fingers[1] == 1:
                    if self.xp is None or self.yp is None:
                        self.xp, self.yp = mx, my
                    if self.tool == "Brush":
                        cv2.line(self.draw_layer, (self.xp, self.yp), (mx, my), self.color, int(self.brush_size))
                    else:
                        cv2.line(self.draw_layer, (self.xp, self.yp), (mx, my), (0,0,0), int(max(self.brush_size*3, 20)))
                    self.xp, self.yp = mx, my
                else:
                    if self.xp is not None and self.yp is not None:
                        self.save_history()
                    self.xp, self.yp = None, None

        with self.lock:
            overlay = cv2.addWeighted(img, 1.0, self.draw_layer, 1.0, 0)

        return VideoFrame.from_ndarray(overlay, format="bgr24")
