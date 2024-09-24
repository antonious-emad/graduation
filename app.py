import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Initialize MediaPipe Pose and Drawing utilities
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Load trained models and scalers
with open("input_scaler.pkl", "rb") as f:
    input_scaler = pickle.load(f)
    
with open("KNN_model.pkl", "rb") as f:
    sklearn_model = pickle.load(f)

# Important landmarks and headers
IMPORTANT_LMS = [
    "NOSE", "LEFT_SHOULDER", "RIGHT_SHOULDER", "RIGHT_ELBOW", 
    "LEFT_ELBOW", "RIGHT_WRIST", "LEFT_WRIST", "LEFT_HIP", "RIGHT_HIP"
]

HEADERS = ["label"] + [f"{lm.lower()}_{coord}" for lm in IMPORTANT_LMS for coord in ["x", "y", "z", "v"]]

# Function to rescale a frame (OpenCV)
def rescale_frame(frame, percent=50):
    width = int(frame.shape[1] * percent / 100)
    height = int(frame.shape[0] * percent / 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

# Function to calculate angles between 3 points
def calculate_angle(point1, point2, point3):
    point1 = np.array(point1)
    point2 = np.array(point2)
    point3 = np.array(point3)

    # Calculate angle
    angle = np.arctan2(point3[1] - point2[1], point3[0] - point2[0]) - np.arctan2(point1[1] - point2[1], point1[0] - point2[0])
    angle = np.abs(angle * 180.0 / np.pi)
    if angle > 180:
        angle = 360 - angle
    return angle

# Function to extract keypoints
def extract_important_keypoints(results, important_landmarks):
    landmarks = results.pose_landmarks.landmark
    data = []
    for lm in important_landmarks:
        keypoint = landmarks[mp_pose.PoseLandmark[lm].value]
        data.append([keypoint.x, keypoint.y, keypoint.z, keypoint.visibility])
    return np.array(data).flatten().tolist()

# Class to handle pose analysis
class BicepPoseAnalysis:
    def __init__(self, side, stage_down_threshold, stage_up_threshold, peak_contraction_threshold, loose_upper_arm_angle_threshold, visibility_threshold):
        self.stage_down_threshold = stage_down_threshold
        self.stage_up_threshold = stage_up_threshold
        self.peak_contraction_threshold = peak_contraction_threshold
        self.loose_upper_arm_angle_threshold = loose_upper_arm_angle_threshold
        self.visibility_threshold = visibility_threshold

        self.side = side
        self.counter = 0
        self.stage = "down"
        self.is_visible = True
        self.detected_errors = {"LOOSE_UPPER_ARM": 0, "PEAK_CONTRACTION": 0}
        self.loose_upper_arm = False
        self.peak_contraction_angle = 1000
        self.peak_contraction_frame = None

    def get_joints(self, landmarks):
        side = self.side.upper()
        joints_visibility = [
            landmarks[mp_pose.PoseLandmark[f"{side}_SHOULDER"].value].visibility,
            landmarks[mp_pose.PoseLandmark[f"{side}_ELBOW"].value].visibility,
            landmarks[mp_pose.PoseLandmark[f"{side}_WRIST"].value].visibility
        ]
        is_visible = all([vis > self.visibility_threshold for vis in joints_visibility])
        self.is_visible = is_visible

        if not is_visible:
            return self.is_visible
        
        self.shoulder = [landmarks[mp_pose.PoseLandmark[f"{side}_SHOULDER"].value].x, landmarks[mp_pose.PoseLandmark[f"{side}_SHOULDER"].value].y]
        self.elbow = [landmarks[mp_pose.PoseLandmark[f"{side}_ELBOW"].value].x, landmarks[mp_pose.PoseLandmark[f"{side}_ELBOW"].value].y]
        self.wrist = [landmarks[mp_pose.PoseLandmark[f"{side}_WRIST"].value].x, landmarks[mp_pose.PoseLandmark[f"{side}_WRIST"].value].y]

        return self.is_visible

    def analyze_pose(self, landmarks, frame):
        self.get_joints(landmarks)
        if not self.is_visible:
            return None, None

        bicep_curl_angle = int(calculate_angle(self.shoulder, self.elbow, self.wrist))
        if bicep_curl_angle > self.stage_down_threshold:
            self.stage = "down"
        elif bicep_curl_angle < self.stage_up_threshold and self.stage == "down":
            self.stage = "up"
            self.counter += 1

        shoulder_projection = [self.shoulder[0], 1]
        ground_upper_arm_angle = int(calculate_angle(self.elbow, self.shoulder, shoulder_projection))

        if ground_upper_arm_angle > self.loose_upper_arm_angle_threshold:
            self.loose_upper_arm = True
            self.detected_errors["LOOSE_UPPER_ARM"] += 1
        else:
            self.loose_upper_arm = False
        
        if self.stage == "up" and bicep_curl_angle < self.peak_contraction_angle:
            self.peak_contraction_angle = bicep_curl_angle
            self.peak_contraction_frame = frame
        elif self.stage == "down":
            if self.peak_contraction_angle != 1000 and self.peak_contraction_angle >= self.peak_contraction_threshold:
                self.detected_errors["PEAK_CONTRACTION"] += 1
            self.peak_contraction_angle = 1000
            self.peak_contraction_frame = None

        return bicep_curl_angle, ground_upper_arm_angle

# Streamlit App
# st.title("Real-Time Bicep Curl Counter & Feedback")

# Start capturing video
cap = cv2.VideoCapture(0)
frame_placeholder = st.empty()

VISIBILITY_THRESHOLD = 0.65
STAGE_UP_THRESHOLD = 90
STAGE_DOWN_THRESHOLD = 120
PEAK_CONTRACTION_THRESHOLD = 60
LOOSE_UPPER_ARM_ANGLE_THRESHOLD = 40
POSTURE_ERROR_THRESHOLD = 0.7
posture = "C"

left_arm_analysis = BicepPoseAnalysis(side="left", stage_down_threshold=STAGE_DOWN_THRESHOLD, stage_up_threshold=STAGE_UP_THRESHOLD, peak_contraction_threshold=PEAK_CONTRACTION_THRESHOLD, loose_upper_arm_angle_threshold=LOOSE_UPPER_ARM_ANGLE_THRESHOLD, visibility_threshold=VISIBILITY_THRESHOLD)
right_arm_analysis = BicepPoseAnalysis(side="right", stage_down_threshold=STAGE_DOWN_THRESHOLD, stage_up_threshold=STAGE_UP_THRESHOLD, peak_contraction_threshold=PEAK_CONTRACTION_THRESHOLD, loose_upper_arm_angle_threshold=LOOSE_UPPER_ARM_ANGLE_THRESHOLD, visibility_threshold=VISIBILITY_THRESHOLD)

feedback_text_placeholder = st.empty()

with mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.8) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = rescale_frame(frame, 100)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        feedback_text = ""

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            landmarks = results.pose_landmarks.landmark
            left_arm_analysis.analyze_pose(landmarks, frame)
            right_arm_analysis.analyze_pose(landmarks, frame)

            row = extract_important_keypoints(results, IMPORTANT_LMS)
            X = pd.DataFrame([row], columns=HEADERS[1:])
            X_scaled = input_scaler.transform(X)
            predicted_class = sklearn_model.predict(X_scaled)[0]
            prediction_probabilities = sklearn_model.predict_proba(X_scaled)[0]

            if max(prediction_probabilities) >= POSTURE_ERROR_THRESHOLD:
                posture = predicted_class

            # feedback_text += f"Posture: {posture}\n"
            feedback_text += f"Left Arm Count: {left_arm_analysis.counter}\n"
            feedback_text += f"Right Arm Count: {right_arm_analysis.counter}\n"

            if left_arm_analysis.loose_upper_arm:
                feedback_text += "Left Arm: Loose Upper Arm!\n"
            if right_arm_analysis.loose_upper_arm:
                feedback_text += "Right Arm: Loose Upper Arm!\n"

        frame_placeholder.image(frame, channels="BGR")
        feedback_text_placeholder.text(feedback_text)

cap.release()

