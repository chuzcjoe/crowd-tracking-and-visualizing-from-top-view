import numpy as np
import cv2
import csv
import math
import time
import math
from utils import pressure
import folium
from utils import normalize, refine_pressure, use_heatmap, expand_pixel, distance, view_angle_pressure
import matplotlib.pyplot as plt 
import seaborn as sns
from imutils.video import FPS
import imutils
import time

# visualize frame X
#frame_num = 150  #95
frame_num = [x for x in range(0,3000,3)] 

# draw radius
draw = False

#draw heatmap
heat = False

plotly_heatmap = False

#draw arrows
arrow = True

#draw line plot
line_plot = False

# write csv
write_csv = False

# define the range for number counting
rad = 50.0

# draw pressure map
pressure_map = True


pressure_map  = None

# write csv file for Turbulence and Speed
save_csv = True


all_positions = []

is_second_frame = False
pre_track_center = None

# functions

OPENCV_OBJECT_TRACKERS = {
    #"csrt": cv2.TrackerCSRT_create,
    "kcf": cv2.TrackerKCF_create,
    "boosting": cv2.TrackerBoosting_create,
    "mil": cv2.TrackerMIL_create,
    "tld": cv2.TrackerTLD_create,
    "medianflow": cv2.TrackerMedianFlow_create,
    "mosse": cv2.TrackerMOSSE_create
  }

# grab the appropriate object tracker using our dictionary of
# OpenCV object tracker objects
tracker = OPENCV_OBJECT_TRACKERS["mosse"]()

# initialize the bounding box coordinates of the object we are going
# to track
initBB = None
         
     

cap = cv2.VideoCapture('videos/Video.mp4')

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 120,    
                       qualityLevel = 0.3, #default 0.3
                       minDistance = 15,  #default 3
                       blockSize = 3 )  #default 3

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),  # default (15,15)
                  maxLevel = 2,   #default 2
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0,255,(200,3))

ret = None
old_frame = None
po = None
old_gray = None

# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)



# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

pressure_map = np.ones_like(old_frame)

#Pmatrix = np.zeros(old_frame.shape)
#Pmatrix[:] = (255,255,255)
#print(old_frame.shape)
#print(Pmatrix.shape)
W, H = old_frame.shape[1], old_frame.shape[0]

#print(Pmatrix[10,10,0] + 3)

cnt = 1
time1 = None
time2 = None

# ids --> id for each person
ids = None

# tracking bounding box center
track_center = None

# indivisual data
indi_pressure = []
indi_position = None
indi_velocity = None
indi_direction = None
indi_time_span = []
indi_plot = True


pressure_per_frame = []
speed_per_frame = []

while(1):
    if cnt % 3 != 0:
         cnt += 1
         #ret, old_frame = cap.read()
         #old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
         #time1 = time.time() 
         #p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
         ret, frame = cap.read()
         continue

    print(cnt)
    ret,frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_copy = frame.copy()


    #print(frame.shape)
    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    #time2 = time.time()
      

    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]

    #------assign id number to each good_new feature-------------
    #if not ids:
    # ids = np.arange(1,good_new.shape[0]+1)

    #print(good_new)


    #----------individual tracking-----------
    # check to see if we are currently tracking an object
    key = cv2.waitKey(1) & 0xFF
 
    # if the 's' key is selected, we are going to "select" a bounding
    # box to track
    if key == ord("s"):
          # select the bounding box of the object we want to track (make
          # sure you press ENTER or SPACE after selecting the ROI)
          initBB = cv2.selectROI("Frame", frame_copy, fromCenter=False,
                          showCrosshair=True)
 
          # start OpenCV object tracker using the supplied bounding box
          # coordinates, then start the FPS throughput estimator as well
          tracker.init(frame_copy, initBB)
          #fps = FPS().start()
    # if the `q` key was pressed, break from the loop
    #------end tracking when the object reaches the boundary
    elif key == ord("q"):
      if indi_plot:
        plt.plot(indi_time_span,indi_pressure)
        plt.xlabel('Frame')
        plt.ylabel('Pressure')
        plt.savefig('indi.png')

      break


    if initBB is not None:
         # grab the new bounding box coordinates of the object
         (success, box) = tracker.update(frame_copy)
 
         # check to see if the tracking was a success
         if success:
             (x, y, w, h) = [int(v) for v in box]
             (f_x,f_y,f_w,f_h) = [v for v in box]
             #print(box)
             track_center = [f_x+f_w//2.0,f_y+f_h//2.0]
             #cv2.circle(frame_copy,(track_center[0],track_center[1]),10,(0,0,255))
             cv2.rectangle(frame_copy, (x, y), (x + w, y + h),
                      (0, 255, 0), 2)

    if pre_track_center is None and track_center is not None:
      pre_track_center = track_center
      is_second_frame = True
      old_gray = frame_gray.copy()
      p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
      cnt += 1
      continue
 
         
    

    
    # calculate the directions, positions and velocity at frame X
    if cnt in frame_num:
      if track_center is not None and is_second_frame:
        #num = good_new.shape[0]
        positions = []
        directions = []
        velocities = []
        #num_count = []
        for (new,old) in zip(good_new,good_old):
             a,b = new.ravel()
             c,d = old.ravel()
             if math.sqrt((a-c)**2+(b-d)**2) < 1:
                   continue
             else:
                   positions.append([a,b])
                   directions.append(list(normalize(np.array((a-c,b-d)))))
                   velocities.append(math.sqrt((a-c)**2+(b-d)**2)/1.0)

        #---------------individual data----------------------
        indi_position = track_center
        indi_velocity = math.sqrt( (track_center[0]-pre_track_center[0])**2+(track_center[1]-pre_track_center[1])**2 )
        indi_direction = list(normalize(np.array((track_center[0]-pre_track_center[0],track_center[1]-pre_track_center[1]))))
        #print(indi_direction)
        pre_track_center = track_center

        in_range = []
        for i, pos in enumerate(positions):
          if 10.0 < distance(track_center,pos) <= rad:
            in_range.append(i)

        #indi_pressure.append(  
        #                 refine_pressure(indi_position,indi_direction,indi_velocity,positions,velocities,directions,in_range) )
        
        indi_pressure.append(  
                         view_angle_pressure(indi_position,indi_direction,indi_velocity,positions,velocities,directions) )
        

        indi_time_span.append(cnt)

        #print(positions)
        #all_positions.extend(positions)

        #for i,pos1 in enumerate(positions):
        #     ct = 0
        #     for j,pos2 in enumerate(positions):
        #          if i == j:
        #               continue
        #          else:
        #              if math.sqrt((pos1[0]-pos2[0])**2+(pos1[1]-pos2[1])**2) <= rad:
        #                      ct += 1
             #print(ct)
        #     num_count.append(ct)
        #print(num_count)
        #print(velocities)
        #cv2.imwrite('viz1.jpg',frame_copy)
        for pos in positions:
            cv2.circle(frame_copy,(pos[0], pos[1]), 5, (0,0,255),1)
            #cv2.putText(frame_copy, str(n), (int(pos[0]-10), int(pos[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), lineType=cv2.LINE_AA)


        #pressure
        #pressure_list = []

        #pressure_heatmap = None
        #pressure_heatmap = np.zeros_like(frame)
        
        #for i,pos1 in enumerate(positions):
        #     in_range = []
        #     for j,pos2 in enumerate(positions):
        #         if i == j:
        #              continue
        #         else:
        #              if math.sqrt((pos1[0]-pos2[0])**2+(pos1[1]-pos2[1])**2) <= rad:
        #                  in_range.append(j)
        #     press = refine_pressure(positions[i],directions[i],velocities[i],positions,velocities,directions,in_range)
        #     pressure_list.append(press)
        #     cv2.rectangle(pressure_heatmap,expand_pixel(13.0,pos1,0.0,float(W),0.0,float(H))[0],expand_pixel(13.0,pos1,0.0,float(W),0.0,float(H))[1],
        #      (0,0,int(press*180.0)),cv2.FILLED)
             #cv2.circle(pressure_heatmap,(pos1[0],pos1[1]),10,(0,0,int(press*80.0)),cv2.FILLED)
             #Pmatrix[ min(int(pos1[1]),319) , min(int(pos1[0]),567), 0:2 ] = (0,0)
             #if Pmatrix[ min(int(pos1[1]),319) , min(int(pos1[0]),567), 2 ] != 255:
             #  Pmatrix[ min(int(pos1[1]),319) , min(int(pos1[0]),567), 2 ] += press
             #else:
             #  Pmatrix[ min(int(pos1[1]),319) , min(int(pos1[0]),567), 2 ] = 10

        

        #pressure_per_frame.append(sum(pressure_list))
        #speed_per_frame.append(sum(velocities)/len(positions))


        

        
        #if cnt == frame_num[-1]:
        #   cv2.imwrite("pressure_map.jpg",pressure_map)
           #img = cv2.add(frame_copy, pressure_map)
        #   img = cv2.add(frame_copy,pressure_map)
        #   cv2.imwrite("press+img.jpg",img)

           # save heatmap

           #img = cv2.resize(img,None, fx = 1, fy = 1, interpolation = cv2.INTER_LINEAR)
           #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
           #cv2.imwrite('heatmap_gray.jpg',gray)
           #im_color = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
           #cv2.imwrite('colormap.jpg',im_color)

           #gray_press = cv2.cvtColor(pressure_map, cv2.COLOR_BGR2GRAY)
           #cv2.imwrite('press_gray.jpg',gray_press)
           #print(len(positions))
           #print(len(all_positions))

           

           #plt.figure(figsize=(6,3))
           #plt.contour(np.flip(gray_press,0),cmap=plt.cm.hot)
           #plt.colorbar()
           #plt.savefig('contourf.jpg')


           #break    
        
    #gray = cv2.cvtColor(pressure_map, cv2.COLOR_BGR2GRAY)          
    
    

    #pressure_map = cv2.add(pressure_map,pressure_heatmap)
    img = cv2.add(frame_copy,mask)
    #img = cv2.add(frame_copy, pressure_map)
    img = cv2.resize(img,(1000,600))
    cnt += 1
    cv2.imshow('frame',img)
    #cv2.imwrite('trajectory.jpg',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    # Now update the previous frame and previous points
    #old_gray = frame_gray.copy()
    #p0 = good_new.reshape(-1,1,2)
    #ret, old_frame = cap.read()

    old_gray = frame_gray.copy()
    p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
    
    #p_ = cv2.goodFeaturesToTrack(frame_gray, mask = None, **feature_params)
    #p_unique = delete_dup(p0,p_)
    #p0 = np.vstack((p0,p_unique))

cv2.destroyAllWindows()
cap.release()

print(len(frame_num),len(pressure_per_frame))

if save_csv:
  with open('data.csv','a') as csv:
                  for i,t,s in zip(frame_num, pressure_per_frame, speed_per_frame):
                     
                          csv.write(str(i)+','+str(t)+','+str(s*0.2)+'\n')


if line_plot:
   import pandas as pd
   #plt.plot(frame_num,pressure_per_frame)
   #plt.show()
   df = pd.DataFrame(dict(frame_num=np.array(frame_num[1:]), Turbulence=np.array(pressure_per_frame)))
   sns_plot = sns.relplot(x='frame_num', y='Turbulence',
            kind='line', marker = True, dashes= False,
            data=df)
   sns_plot.savefig("output.png")


