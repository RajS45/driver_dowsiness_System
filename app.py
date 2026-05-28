from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import base64
import mediapipe as mp
from utils import calculate_ear, calculate_mar
from config import *

app = Flask(__name__)
# SocketIO allows real-time, bi-directional communication
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize MediaPipe once globally instead of inside a loop
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=MIN_DETECTION_CONFIDENCE,
    min_tracking_confidence=MIN_TRACKING_CONFIDENCE
)

LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
MOUTH_INDICES = [78, 81, 13, 311, 308, 402, 14, 82]

# Shared tracking state across incoming frames
state = {"EYE_COUNTER": 0, "MOUTH_COUNTER": 0}

@app.route('/')
def index():
    return render_template('index.html')

# Listen for incoming image frames from the web browser
@socketio.on('process_frame')
def handle_frame(data):
    global state
    try:
        # Decode the base64 image string sent by the browser JavaScript
        encoded_data = data['image'].split(',')[1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        drowsy = False
        yawn = False
        avg_ear = 0.0
        avg_mar = 0.0

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                coords = np.array([(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark])
                
                left_ear = calculate_ear(coords[LEFT_EYE_INDICES])
                right_ear = calculate_ear(coords[RIGHT_EYE_INDICES])
                avg_ear = (left_ear + right_ear) / 2.0
                avg_mar = calculate_mar(coords[MOUTH_INDICES])

                # Drowsiness Frame Counting Thresholds
                if avg_ear < EYE_AR_THRESH:
                    state["EYE_COUNTER"] += 1
                    if state["EYE_COUNTER"] >= EYE_AR_CONSEC_FRAMES:
                        drowsy = True
                else:
                    state["EYE_COUNTER"] = 0

                # Yawn Frame Counting Thresholds
                if avg_mar > MOUTH_AR_THRESH:
                    state["MOUTH_COUNTER"] += 1
                    if state["MOUTH_COUNTER"] >= YAWN_CONSEC_FRAMES:
                        yawn = True
                else:
                    state["MOUTH_COUNTER"] = 0

        # Send calculations immediately back to the specific client browser
        emit('response_metrics', {
            'ear': round(avg_ear, 2),
            'mar': round(avg_mar, 2),
            'drowsy': drowsy,
            'yawn': yawn
        })
    except Exception as e:
        print(f"Frame processing error: {e}")

if __name__ == '__main__':
    # Use socketio.run instead of app.run to support WebSocket connections
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)