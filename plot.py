import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import math
from scipy.interpolate import griddata

def dis(x1,x2):
    return math.sqrt((x1[0]-x2[0])**2+(x1[1]-x2[1])**2)

x = np.arange(100)
y = np.arange(100)

z = np.zeros((100))
z[66] = 0.5
z[17] = 0.88
z[56] = 0.12
z[89] = 0.62

def hmplot(x,y,z):
    xi = np.linspace(-1,101,100)
    yi = np.linspace(-1,101,100)
    zi = griddata((x, y), z, (xi[None,:], yi[:,None]), method='cubic')
    CS = plt.contour(xi,yi,zi,15,linewidths=0.5,colors='k')
    CS = plt.contourf(xi,yi,zi,15,cmap=plt.cm.jet)
    plt.colorbar() # draw colorbar
    plt.show()

hmplot(x,y,z)

#f,ax=plt.subplots(figsize=(10,5))

#thre = 30
#decre = 1
#n = 100
#c = 50

#a = np.zeros((n, n))
#a[c,c] = 200
#center = [c,c]

#for i in range(n):
#    for j in range(n):
#        d = dis([i,j],center)
#        if d > thre:
#           a[i,j] = 0
#        else:
#           a[i,j] = a[c,c] - d*decre

#pic = sns.heatmap(a, annot=True, fmt="d", ax=ax)

#f.savefig('sns_heatmap_eg2.jpg')
#plt.imshow(a, cmap='viridis', interpolation='nearest')
#plt.show()
