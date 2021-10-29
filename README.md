# Running Form
This project will look at your running form using the Mediapipe library from Google. It takes pose landmarks and calculates angles of joints in relation to each other. 

![alt text](https://google.github.io/mediapipe/images/mobile/pose_tracking_full_body_landmarks.png)

Using this, the program can look at the form of professional athletes and calculate what the ideal running form is. 
When someone takes a video of themself running, they can pass it through the program to see how good their running form is.

The joints are coordinates on a numpy array. The angles are calculated with trignometery after finding the length of the joints with the coordinates. 

![alt text](https://www.mathsisfun.com/algebra/images/adjacent-opposite-hypotenuse.svg)
