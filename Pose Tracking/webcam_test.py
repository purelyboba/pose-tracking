# imports
import mediapipe as mp
import cv2
import numpy as np

# setting up drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# variables to keep track of number of reps
reps = 0
up = False

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
cap = cv2.VideoCapture(0)

with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:

    while cap.isOpened():
        ret, frame = cap.read()

        frame = cv2.resize(frame, (640, 480))

        # detection
        results = pose.process(frame)

        # extract landmarks
        try:
            # array of 33 landmarks (see pose_tracking_full_body_landmarks.png)
            landmarks = results.pose_landmarks.landmark
            # print(landmarks)
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            print(calculate_angle(shoulder, elbow, wrist))
            angle = calculate_angle(shoulder, elbow, wrist)

            if angle >= 120:
                up = False
            
            if angle < 120 and up == False:
                up = True
                reps += 1
                print(reps)
            
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


cap.release()
cv2.destroyAllWindows()