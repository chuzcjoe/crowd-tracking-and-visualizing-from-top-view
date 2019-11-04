# untils for visualization
import numpy as np
import cv2
import pandas as pd
import scipy
import math

import numpy as np

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)) * 180 / math.pi


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

def generateBaseMap(default_location=[200, 200], default_zoom_start=12):
    base_map = folium.Map(location=default_location, control_scale=True, zoom_start=default_zoom_start)
    return base_map


def use_heatmap(image, box_centers):
    import heatmap
    hm = heatmap.Heatmap()
    box_centers = [(i, image.shape[0]-j) for i, j in box_centers]
    img = hm.heatmap(box_centers, dotsize=150, size=(image.shape[1], image.shape[0]), opacity=128, area=((0, 0), (image.shape[1], image.shape[0])))
    img.save('hm.png')


def delete_dup(array1,array2):
    n1 = array1.shape[0]
    n2 = array2.shape[0]
    idx = []
    for i in range(n1):
          for j in range(n2):
                if (array1[i] == array2[j]).all():
                    idx.append(j)
    return np.delete(array2,idx,0)



def refine_pressure(center_pos,center_dir,center_vel,ps,vs,ds,index):
    if not index:
       return 0

    positions = [ps[i] for i in index]
    velocities = [vs[i] for i in index]
    directions = [ds[i] for i in index]
    T = []
    center_vector = [center_dir[0]*center_vel,center_dir[1]*center_vel]

    Vectors = []
    pressure = []
    for d,v in zip(directions,velocities):
        Vectors.append([d[0]*v,d[1]*v])

    for p,vec in zip(positions,Vectors):
        pressure.append(math.sqrt((vec[0]-center_vector[0])**2+(vec[1]-center_vector[1])**2) / math.sqrt((center_pos[0]-p[0])**2+(center_pos[1]-p[1])**2))
    return sum(pressure)

def pressure(center_pos, center_dir, ps, vs, ds, index, weights):

    if not index:
        return 0
    positions = [ps[i] for i in index]
    velocities = [vs[i] for i in index]
    directions = [ds[i] for i in index]
    densities = []
    speeds = []
    turbs = []
    SUM = []
    for s in velocities:
        speeds.append(math.exp(-1/2*s))
  
    for pos in positions:
        densities.append(math.exp(-1/2*(math.sqrt((center_pos[0]-pos[0])**2+(center_pos[1]-pos[1])**2))))
    
    for d in directions:
        turbs.append(1.0/180.0 * angle_between(center_dir,d))

    for d, s, t in zip(densities,speeds, turbs):
        SUM.append(weights[0]*d+weights[1]*s+weights[2]*t)
   
    return sum(SUM)





