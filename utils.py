import numpy as np
from scipy.spatial import distance as dist

def calculate_ear(eye_landmarks):
    """
    Calculates Eye Aspect Ratio (EAR) given 6 landmark points around an eye.
    """
    # Compute the euclidean distances between the two sets of vertical eye landmarks
    A = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
    B = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
    
    # Compute the euclidean distance between the horizontal eye landmark
    C = dist.euclidean(eye_landmarks[0], eye_landmarks[3])
    
    # Compute EAR
    ear = (A + B) / (2.0 * C)
    return ear

def calculate_mar(mouth_landmarks):
    """
    Calculates Mouth Aspect Ratio (MAR) using key internal lip landmarks.
    """
    # Vertical distances between upper and lower lips
    A = dist.euclidean(mouth_landmarks[1], mouth_landmarks[7])
    B = dist.euclidean(mouth_landmarks[2], mouth_landmarks[6])
    C = dist.euclidean(mouth_landmarks[3], mouth_landmarks[5])
    
    # Horizontal distance between corners of the mouth
    D = dist.euclidean(mouth_landmarks[0], mouth_landmarks[4])
    
    # Compute MAR
    mar = (A + B + C) / (3.0 * D)
    return mar