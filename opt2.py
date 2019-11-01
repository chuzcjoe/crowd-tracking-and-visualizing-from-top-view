import numpy as np
import cv2
import csv
import time
import math

# visualize frame X
frame_num = 95

# draw radius
draw = True

#draw arrows
arrow = True

# define the range for number counting
rad = 50

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm


def delete_dup(array1,array2):
    n1 = array1.shape[0]
    n2 = array2.shape[0]
    idx = []
    for i in range(n1):
          for j in range(n2):
                if (array1[i] == array2[j]).all():
                    idx.append(j)
    return np.delete(array2,idx,0)
                      
     

cap = cv2.VideoCapture('Video.mp4')

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 50,
                       qualityLevel = 0.3,
                       minDistance = 5,
                       blockSize = 5 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
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

while(1):
    if cnt % 5 != 0:
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
    #print(good_new.shape)

    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
   
    # calculate the directions, positions and velocity at frame X
    if cnt == frame_num:
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
                   velocities.append(math.sqrt((a-c)**2+(b-d)**2))

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
        print(num_count)
        print(directions)
        cv2.imwrite('viz1.jpg',frame_copy)
        for pos,n in zip(positions,num_count):
            cv2.circle(frame_copy,(pos[0], pos[1]), 5, (0,255,0),1)
            cv2.putText(frame_copy, str(n), (int(pos[0]-10), int(pos[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), lineType=cv2.LINE_AA)

        if draw:
            for pos in positions:
                cv2.circle(frame_copy, (pos[0],pos[1]),rad,(255,0,0),1)

        if arrow:
            for pos,di in zip(positions,directions):
                cv2.arrowedLine(frame_copy, (int(pos[0]),int(pos[1])),(int(pos[0]+30.0*di[0]),int(pos[1]+30.0*di[1])),(0,0,255),1)
        cv2.imwrite('viz2.jpg',frame_copy)   

        with open('crowd'+str(cnt)+'.csv','a') as csv:
             for pos,di,v,n in zip(positions,directions,velocities,num_count):
                     
                     csv.write(str(pos[0])+','+str(pos[1])+','+str(n)+','+str(v)+','+str(di[0])+','+str(di[1])+'\n')
        break        


    img = cv2.add(frame,mask)
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
