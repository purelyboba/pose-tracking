# imports
import mediapipe as mp
import cv2
import numpy as np
import matplotlib.pyplot as plt

# setting up drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# list to keep track of the angles
angle_1 = []
angle_2 = []
angle_3 = []
angle_4 = []
angle_5 = []
angle_6 = []
angle_7 = []
angle_8 = []
angle_9 = []
angle_10 = []

num_of_frames = 0

# uses trignometry to calculate the angle when given three points
def calculate_angle(a, b, c):
    a = np.array(a) # shoulder
    b = np.array(b) # elbow
    c = np.array(c) # wrist

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180.0:
        angle = 360-angle
    
    return angle

# set up the webcam feed
# cap = cv2.VideoCapture('wired_test.mp4')
cap = cv2.VideoCapture("videos/plank.mp4")

with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:

    while cap.isOpened():
        ret, frame = cap.read()

        frame = cv2.resize(frame, (384, 216))

        # detection
        results = pose.process(frame)

        # extract landmarks
        try:
            # array of 33 landmarks (see pose_tracking_full_body_landmarks.png)
            landmarks = results.pose_landmarks.landmark
            # print(landmarks)
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x, landmarks[mp_pose.PoseLandmark.NOSE.value].y]

            left_ankle_knee_hip_angle = calculate_angle(left_ankle, left_knee, left_hip)
            right_ankle_knee_hip_angle = calculate_angle(right_ankle, right_knee, right_hip)
            left_knee_hip_shoulder_angle = calculate_angle(left_knee, left_hip, left_shoulder)
            right_knee_hip_shoulder_angle = calculate_angle(right_knee, right_hip, right_shoulder)
            left_hip_shoulder_elbow_angle = calculate_angle(left_hip, left_shoulder, left_elbow)
            right_hip_shoulder_elbow_angle = calculate_angle(right_hip, right_shoulder, right_elbow)
            left_shoulder_elbow_wrist_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            right_shoulder_elbow_wrist_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
            left_hip_shoulder_nose_angle = calculate_angle(left_hip, left_shoulder, nose)
            right_hip_shoulder_nose_angle = calculate_angle(right_hip, right_shoulder, nose)

            print("Left: ", 
                  left_ankle_knee_hip_angle, 
                  left_knee_hip_shoulder_angle, 
                  left_hip_shoulder_elbow_angle, 
                  left_shoulder_elbow_wrist_angle, 
                  left_hip_shoulder_nose_angle)
            
            print("Right: ",
                  right_ankle_knee_hip_angle, 
                  right_knee_hip_shoulder_angle, 
                  right_hip_shoulder_elbow_angle, 
                  right_shoulder_elbow_wrist_angle, 
                  right_hip_shoulder_nose_angle)

            print("-----------------------------------------------------------------------------------------------------")

            angle_1.append(left_ankle_knee_hip_angle)
            angle_2.append(right_ankle_knee_hip_angle)
            angle_3.append(left_knee_hip_shoulder_angle)
            angle_4.append(right_knee_hip_shoulder_angle)
            angle_5.append(left_hip_shoulder_elbow_angle)
            angle_6.append(right_hip_shoulder_elbow_angle)
            angle_7.append(left_shoulder_elbow_wrist_angle)
            angle_8.append(right_shoulder_elbow_wrist_angle)
            angle_9.append(left_hip_shoulder_nose_angle)
            angle_10.append(right_hip_shoulder_nose_angle)
            
            num_of_frames += 1

            # print(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value])

        except:
            pass

        # draw detections on webcam feed (frame)
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
            )

        # show detections
        cv2.imshow('Pose tracking', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

        if ret == False:
            break

cap.release()
cv2.destroyAllWindows()

average_1 = sum(angle_1)/len(angle_1)
average_2 = sum(angle_2)/len(angle_2)
average_3 = sum(angle_3)/len(angle_3)
average_4 = sum(angle_4)/len(angle_4)
average_5 = sum(angle_5)/len(angle_5)
average_6 = sum(angle_6)/len(angle_6)
average_7 = sum(angle_7)/len(angle_7)
average_8 = sum(angle_8)/len(angle_8)
average_9 = sum(angle_9)/len(angle_9)
average_10 = sum(angle_10)/len(angle_10)

print(average_1, average_2, average_3, average_4, average_5, average_6, average_7, average_8, average_9, average_10)
