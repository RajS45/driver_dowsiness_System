# Eye Aspect Ratio (EAR) Thresholds
EYE_AR_THRESH = 0.21         # Balanced for true micro-sleeps, ignores fast blinks
EYE_AR_CONSEC_FRAMES = 14    # ~450ms at 30 FPS (completely clears natural human blinks)

# Mouth Aspect Ratio (MAR) Thresholds (Yawning)
MOUTH_AR_THRESH = 0.68       
YAWN_CONSEC_FRAMES = 15      

# MediaPipe configuration optimized for speed
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# Audio Configuration
ALARM_VOLUME = 1.0           # MAX VOLUME (Range: 0.0 to 1.0)