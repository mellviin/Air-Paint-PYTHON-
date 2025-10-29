"""
AR Air Paint - Single Window Version (Fixed Canvas & Improved Shape Logic)
Features:
- Draw directly on live video feed
- Brush types: normal, dotted, calligraphy
- Adjustable brush/eraser size
- Color palette
- Shape insertion: rectangle, circle, line (with preview)
- Undo / Redo
- Save / Load
- Gesture-based controls using MediaPipe
"""

import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import os

# Config
CAM_WIDTH = 1280
CAM_HEIGHT = 720
HISTORY_LIMIT = 20

# MediaPipe hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Undo/redo
history = deque(maxlen=HISTORY_LIMIT)
redo_stack = []

# Drawing state
xp, yp = None, None
color = (0, 0, 255)
brush_size = 8
eraser_size = 60
tool = 'brush'
brush_type = 'normal'
shape_type = None
shape_start = None
drawing_shape = False

# UI
PALETTE = [
    {'name': 'black', 'bgr': (0, 0, 0), 'rect': (20, 20, 100, 100)},
    {'name': 'red', 'bgr': (0, 0, 255), 'rect': (130, 20, 210, 100)},
    {'name': 'green', 'bgr': (0, 255, 0), 'rect': (240, 20, 320, 100)},
    {'name': 'blue', 'bgr': (255, 0, 0), 'rect': (350, 20, 430, 100)},
    {'name': 'yellow', 'bgr': (0, 255, 255), 'rect': (460, 20, 540, 100)},
    {'name': 'eraser', 'bgr': (255, 255, 255), 'rect': (570, 20, 650, 100)}
]

SIZE_UI = [
    {'name': 'S', 'size': 5, 'rect': (20, 120, 100, 200)},
    {'name': 'M', 'size': 12, 'rect': (130, 120, 210, 200)},
    {'name': 'L', 'size': 25, 'rect': (240, 120, 320, 200)}
]

SHAPE_UI = [
    {'name': 'rect', 'rect': (20, 220, 120, 300)},
    {'name': 'circle', 'rect': (140, 220, 240, 300)},
    {'name': 'line', 'rect': (260, 220, 360, 300)}
]

# Functions
def save_history(layer):
    history.append(layer.copy())
    redo_stack.clear()

def undo(layer):
    global history, redo_stack
    if len(history) > 1:
        redo_stack.append(history.pop())
        layer[:] = history[-1].copy()

def redo(layer):
    global history, redo_stack
    if redo_stack:
        layer[:] = redo_stack.pop()
        history.append(layer.copy())

def draw_ui(frame):
    # Draw color palette
    for item in PALETTE:
        x1, y1, x2, y2 = item['rect']
        cv2.rectangle(frame, (x1,y1), (x2,y2), item['bgr'], -1)
        cv2.rectangle(frame, (x1,y1), (x2,y2), (50,50,50), 2)
        cv2.putText(frame, item['name'], (x1+5, y2-8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0,0,0) if item['name']=='eraser' else (255,255,255), 2)
    # Brush sizes
    for item in SIZE_UI:
        x1,y1,x2,y2 = item['rect']
        cv2.rectangle(frame,(x1,y1),(x2,y2),(220,220,220),-1)
        cv2.rectangle(frame,(x1,y1),(x2,y2),(80,80,80),2)
        cv2.putText(frame, item['name'], (x1+10, y2-10), cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,0),2)
        cx = x1 + (x2-x1)//2
        cy = y1 + (y2-y1)//2
        cv2.circle(frame,(cx,cy),item['size'],(0,0,0),-1)
    # Shape buttons
    for item in SHAPE_UI:
        x1,y1,x2,y2 = item['rect']
        cv2.rectangle(frame,(x1,y1),(x2,y2),(200,200,200),-1)
        cv2.rectangle(frame,(x1,y1),(x2,y2),(80,80,80),2)
        cv2.putText(frame,item['name'],(x1+8,y2-8),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),2)

def fingers_up(hand_landmarks):
    tips=[4,8,12,16,20]
    fingers=[]
    try: fingers.append(1 if hand_landmarks.landmark[tips[0]].x < hand_landmarks.landmark[tips[0]-1].x else 0)
    except: fingers.append(0)
    for id in range(1,5):
        try: fingers.append(1 if hand_landmarks.landmark[tips[id]].y < hand_landmarks.landmark[tips[id]-2].y else 0)
        except: fingers.append(0)
    return fingers

# Main Loop
cap = cv2.VideoCapture(0)
cap.set(3,CAM_WIDTH)
cap.set(4,CAM_HEIGHT)
draw_layer = None

print("AR Air Paint - Press 'q' to quit, 's' to save, 'l' to load, 'u' undo, 'r' redo, 'c' clear, '1/2/3' brush types")

while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame,1)
    if draw_layer is None:
        draw_layer = np.zeros_like(frame)

    h,w,_ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(frame_rgb)

    canvas = frame.copy()
    all_fingers_down = False

    if res.multi_hand_landmarks:
        for handLms in res.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
            index = handLms.landmark[8]
            mx,my = int(index.x*w), int(index.y*h)
            fingers = fingers_up(handLms)
            all_fingers_down = sum(fingers)==0

            # Selection mode (Index + Middle)
            if fingers[1]==1 and fingers[2]==1:
                xp,yp=None,None
                for item in PALETTE:
                    x1,y1,x2,y2 = item['rect']
                    if x1<=mx<=x2 and y1<=my<=y2:
                        color = item['bgr']
                        tool = 'eraser' if item['name']=='eraser' else 'brush'
                        save_history(draw_layer)
                for item in SIZE_UI:
                    x1,y1,x2,y2 = item['rect']
                    if x1<=mx<=x2 and y1<=my<=y2:
                        brush_size = item['size']
                        eraser_size = max(brush_size*4,30)
                        save_history(draw_layer)
                for item in SHAPE_UI:
                    x1,y1,x2,y2 = item['rect']
                    if x1<=mx<=x2 and y1<=my<=y2:
                        shape_type = item['name']
                        tool = 'shape'
                        shape_start = None
                        drawing_shape=False
                        save_history(draw_layer)

            # Only switch tool if NOT currently using brush or eraser manually
            if tool not in ['brush', 'eraser']:
                if sum(fingers) == 2 and fingers[1] == 1 and fingers[2] == 1:  # rectangle
                    tool = 'shape'; shape_type = 'rect'
                elif fingers[1] == 1 and fingers[0] == 1:  # circle
                    tool = 'shape'; shape_type = 'circle'
                elif fingers[1] == 1 and sum(fingers) == 1:  # line
                    tool = 'shape'; shape_type = 'line'

            # Drawing
            if fingers[1]==1:
                if xp is None or yp is None:
                    xp,yp = mx,my
                    if tool=='shape' and not drawing_shape:
                        shape_start = (mx,my)
                        drawing_shape=True

                if tool=='brush':
                    if brush_type=='normal':
                        cv2.line(draw_layer,(xp,yp),(mx,my),color,brush_size)
                    elif brush_type=='dotted':
                        dist=int(np.hypot(mx-xp,my-yp))
                        steps=max(dist//(brush_size*2),1)
                        for i in range(steps+1):
                            t=i/steps
                            xi=int(xp+(mx-xp)*t)
                            yi=int(yp+(my-yp)*t)
                            cv2.circle(draw_layer,(xi,yi),max(1,brush_size//2),color,-1)
                    elif brush_type=='calligraphy':
                        angle=np.arctan2(my-yp,mx-xp)
                        ox=int(np.cos(angle+np.pi/2)*brush_size/2)
                        oy=int(np.sin(angle+np.pi/2)*brush_size/2)
                        pts=np.array([[xp-ox,yp-oy],[xp+ox,yp+oy],[mx+ox,my+oy],[mx-ox,my-oy]])
                        cv2.fillConvexPoly(draw_layer,pts,color)
                    xp,yp=mx,my

                elif tool=='eraser':
                    cv2.line(draw_layer,(xp,yp),(mx,my),(0,0,0),eraser_size)
                    xp,yp=mx,my

                elif tool=='shape' and drawing_shape:
                    temp = draw_layer.copy()
                    x0,y0 = shape_start
                    x1,y1 = mx,my
                    if shape_type=='rect': cv2.rectangle(temp,(x0,y0),(x1,y1),color,brush_size)
                    elif shape_type=='circle': cv2.circle(temp,(x0,y0),int(np.hypot(x1-x0,y1-y0)),color,brush_size)
                    elif shape_type=='line': cv2.line(temp,(x0,y0),(x1,y1),color,brush_size)
                    canvas = cv2.addWeighted(frame,1,temp,1,0)
            else:
                if tool=='shape' and drawing_shape and all_fingers_down:
                    x0,y0 = shape_start
                    x1,y1 = mx,my
                    if shape_type=='rect': cv2.rectangle(draw_layer,(x0,y0),(x1,y1),color,brush_size)
                    elif shape_type=='circle': cv2.circle(draw_layer,(x0,y0),int(np.hypot(x1-x0,y1-y0)),color,brush_size)
                    elif shape_type=='line': cv2.line(draw_layer,(x0,y0),(x1,y1),color,brush_size)
                    drawing_shape=False
                    shape_start=None
                    save_history(draw_layer)
                xp,yp=None,None

    # Overlay drawing layer
    canvas = cv2.addWeighted(canvas,1,draw_layer,1,0)

    # Draw UI
    draw_ui(canvas)

    # Tool info
    cv2.putText(canvas,f'Tool: {tool} | Brush: {brush_type} | Size: {brush_size}',
                (20,CAM_HEIGHT-20),cv2.FONT_HERSHEY_SIMPLEX,0.7,(50,50,50),2)

    cv2.imshow("AR Air Paint",canvas)

    key=cv2.waitKey(1)&0xFF
    if key==27 or key==ord('q'): break
    elif key==ord('s'): cv2.imwrite('drawing.png',draw_layer); print("Saved drawing.png")
    elif key==ord('l') and os.path.exists('drawing.png'):
        loaded=cv2.imread('drawing.png')
        if loaded is not None: draw_layer=loaded.copy(); save_history(draw_layer); print("Loaded drawing.png")
    elif key==ord('u'): undo(draw_layer)
    elif key==ord('r'): redo(draw_layer)
    elif key==ord('c'): draw_layer = np.zeros_like(frame); save_history(draw_layer)
    elif key==ord('1'): brush_type='normal'
    elif key==ord('2'): brush_type='dotted'
    elif key==ord('3'): brush_type='calligraphy'

cap.release()
cv2.destroyAllWindows()
