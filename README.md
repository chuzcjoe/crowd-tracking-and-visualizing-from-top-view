# crowd-tracking-and-visualizing-from-top-view

optical flow is used for moving object detection in the video shot by a drone camera.

requirements:
1. numpy
2. opencv3

For each frame, each object's moving direction, speed, number of people in the surrounding areas could be calculated and visualized

![image](https://github.com/chuzcjoe/crowd-tracking-and-visualizing-from-top-view/raw/master/viz1.jpg)
![image](https://github.com/chuzcjoe/crowd-tracking-and-visualizing-from-top-view/raw/master/viz2.jpg)


We use speed, velocity and direction to define the Turbulence level for each extracted frame.
![image](https://github.com/chuzcjoe/crowd-tracking-and-visualizing-from-top-view/raw/master/output.png)
