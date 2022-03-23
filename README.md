# Running Form
This project will look at your yoga pose in relation to an instruction video using the Mediapipe library from Google. It takes pose landmarks and calculates angles of joints in relation to each other. 

![alt text](https://google.github.io/mediapipe/images/mobile/pose_tracking_full_body_landmarks.png)

Using this, a person can compare their yoga pose to that of an instruction video, and adjust accordingly.

The joints are coordinates on a numpy array. The angles are calculated with trignometery after finding the length of the joints with the coordinates. 

![alt text](https://www.mathsisfun.com/algebra/images/adjacent-opposite-hypotenuse.svg)

To do:
- create system to analyze differences between your pose and that of video
- create a front-end user interface (app and website)
- deploy on web, google play store, and app store (hopefully)
