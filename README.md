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

Pressure Map for showing the geospatial distribution of the pressure value.
![image](https://github.com/chuzcjoe/crowd-tracking-and-visualizing-from-top-view/raw/master/interpolate.jpg)
![image](https://github.com/chuzcjoe/crowd-tracking-and-visualizing-from-top-view/raw/master/colab_contour.jpg)
Moreover, individual tracking is added and individual pressure is measured.
![image](https://github.com/chuzcjoe/crowd-tracking-and-visualizing-from-top-view/raw/master/indi_tracki.png)
![image](https://github.com/chuzcjoe/crowd-tracking-and-visualizing-from-top-view/raw/master/indi.png)
