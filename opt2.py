import numpy as np
import cv2
import csv
import time
import math
from utils import pressure
import folium
from utils import normalize, refine_pressure, use_heatmap
import matplotlib.pyplot as plt 

# visualize frame X
#frame_num = 150  #95
frame_num = [x for x in range(10,300,10)] 

# draw radius
draw = False

#draw heatmap
heat = False

#draw arrows
arrow = True

#draw line plot
line_plot = True

# write csv
write_csv = False

# define the range for number counting
rad = 50


# functions

                      
     

cap = cv2.VideoCapture('videos/Video_3.MOV')

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 60,    
                       qualityLevel = 0.25, #default 0.3
                       minDistance = 10,  #default 3
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

cnt = 1
time1 = None
time2 = None

pressure_per_frame = []

while(1):
    if cnt % 10 != 0:
         #ret, old_frame = cap.read()
         #old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
         #time1 = time.time() 
         #p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
         ret, frame = cap.read()
         cnt += 1
         continue

    
    print(cnt)   
    ret,frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_copy = frame.copy()

    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    #time2 = time.time()
      

    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]
    

    # draw the tracks
    #for i,(new,old) in enumerate(zip(good_new,good_old)):
    #    a,b = new.ravel()
    #    c,d = old.ravel()
    #    mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
    #    frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
   
    # calculate the directions, positions and velocity at frame X
    if cnt in frame_num:
        num = good_new.shape[0]
        positions = []
        directions = []
        velocities = []
        num_count = []
        for (new,old) in zip(good_new,good_old):
             a,b = new.ravel()
             c,d = old.ravel()
             if math.sqrt((a-c)**2+(b-d)**2) < 1:
                   continue
             else:
                   positions.append([a,b])
                   directions.append(list(normalize(np.array((a-c,b-d)))))
                   velocities.append(math.sqrt((a-c)**2+(b-d)**2)/1.0)

        for i,pos1 in enumerate(positions):
             ct = 0
             for j,pos2 in enumerate(positions):
                  if i == j:
                       continue
                  else:
                      if math.sqrt((pos1[0]-pos2[0])**2+(pos1[1]-pos2[1])**2) <= rad:
                              ct += 1
             #print(ct)
             num_count.append(ct)
        #print(num_count)
        #print(velocities)
        cv2.imwrite('viz1.jpg',frame_copy)
        for pos,n in zip(positions,num_count):
            cv2.circle(frame_copy,(pos[0], pos[1]), 5, (0,255,0),1)
            #cv2.putText(frame_copy, str(n), (int(pos[0]-10), int(pos[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), lineType=cv2.LINE_AA)


        #pressure
        pressure_list = []
        for i,pos1 in enumerate(positions):
             in_range = []
             for j,pos2 in enumerate(positions):
                 if i == j:
                      continue
                 else:
                      if math.sqrt((pos1[0]-pos2[0])**2+(pos1[1]-pos2[1])**2) <= rad:
                          in_range.append(j)
             pressure_list.append(refine_pressure(positions[i],directions[i],velocities[i],positions,velocities,directions,in_range))

        pressure_per_frame.append(sum(pressure_list))

        for i,pos in enumerate(positions):
            cv2.putText(frame_copy,  str('%.3f' % pressure_list[i]), (int(pos[0]-10), int(pos[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), lineType=cv2.LINE_AA)


        if draw:
            for pos in positions:
                cv2.circle(frame_copy, (pos[0],pos[1]),rad,(255,0,0),1)

        if arrow:
            for pos,di in zip(positions,directions):
                cv2.arrowedLine(frame_copy, (int(pos[0]),int(pos[1])),(int(pos[0]+30.0*di[0]),int(pos[1]+30.0*di[1])),(0,0,255),1)
        cv2.imwrite('viz2.jpg',frame_copy)   
         
        # write csv
        if write_csv:
             with open('crowd'+str(cnt)+'.csv','a') as csv:
                  for pos,di,v,n in zip(positions,directions,velocities,num_count):
                     
                          csv.write(str(pos[0])+','+str(pos[1])+','+str(n)+','+str(v)+','+str(di[0])+','+str(di[1])+'\n')

        
        # draw heatmap
        if heat:
           use_heatmap(frame_copy, positions)
           hm = cv2.imread('hm.png')
           alpha = 0.3 # transparency
           overlay = frame.copy()
           #cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1) 
           cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame) 
           cv2.addWeighted(hm, alpha, frame, 1-alpha, 0, frame) 
           cv2.imwrite('heatmap.jpg',frame)
        
        if cnt == frame_num[-1]:
           break    
        
               


    img = cv2.add(frame_copy,mask)
    img = cv2.resize(img,(1200,800))
    cnt += 1
    cv2.imshow('frame',img)
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


if line_plot:
   plt.plot(frame_num,pressure_per_frame)
   plt.show()


