# crowd-tracking-and-visualizing-from-top-view

optical flow is used for moving object detection in the video shot by a drone camera.

requirements:
1. numpy
2. opencv3
3. math
4. seaborn
5. heatmap
6. matplotlib
7. scipy

For each frame, each object's moving direction, speed, number of people in the surrounding areas could be calculated and visualized

![image](https://github.com/chuzcjoe/crowd-tracking-and-visualizing-from-top-view/raw/master/viz1.jpg)
![image](https://github.com/chuzcjoe/crowd-tracking-and-visualizing-from-top-view/raw/master/viz2.jpg)


We use speed, velocity and direction to define the Turbulence level for each extracted frame.
![image](https://github.com/chuzcjoe/crowd-tracking-and-visualizing-from-top-view/raw/master/output.png)

Our math model for computing the Turbulence level allows us to create pressure map for better visually showing the crowd movement.
![image](https://github.com/chuzcjoe/crowd-tracking-and-visualizing-from-top-view/raw/master/press%2Bimg.jpg)
![image](https://github.com/chuzcjoe/crowd-tracking-and-visualizing-from-top-view/blob/master/press.jpg)
