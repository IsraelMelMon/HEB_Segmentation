import cv2
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import numpy as np 

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors

from matplotlib.colors import hsv_to_rgb

light_orange =  (0, 52, 152)
dark_orange = (179, 255, 227)


lo_square = np.full((10, 10, 3), light_orange, dtype=np.uint8) / 255.0
do_square = np.full((10, 10, 3), dark_orange, dtype=np.uint8) / 255.0

#plt.subplot(2, 1, 1)
#plt.imshow((lo_square))
#plt.subplot(2, 1, 2)
#plt.imshow((do_square))
#plt.show()

"""
image = mpimg.imread('test00001.jpg')
plt.imshow(image)
plt.show()
"""

image = cv2.imread('test00002.jpg',cv2.COLOR_BGR2RGB)
#image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
plt.imshow(image)
plt.show()
"""
r, g, b = cv2.split(image)
fig = plt.figure()
axis = fig.add_subplot(1, 1, 1, projection="3d")

pixel_colors = image.reshape((np.shape(image)[0]*np.shape(image)[1], 3))
norm = colors.Normalize(vmin=-1.,vmax=1.)
norm.autoscale(pixel_colors)
pixel_colors = norm(pixel_colors).tolist()

axis.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors=pixel_colors, marker=".")
axis.set_xlabel("Red")
axis.set_ylabel("Green")
axis.set_zlabel("Blue")
plt.show()
"""
pixel_colors = image.reshape((np.shape(image)[0]*np.shape(image)[1], 3))
norm = colors.Normalize(vmin=-1.,vmax=1.)
norm.autoscale(pixel_colors)
pixel_colors = norm(pixel_colors).tolist()
hsv_nemo = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
h, s, v = cv2.split(hsv_nemo)
fig = plt.figure()
axis = fig.add_subplot(1,1,1,projection="3d")
#c = np.random.rand(len(h))
axis.scatter(h.flatten(), s.flatten(), v.flatten(), c=pixel_colors, marker=".")
#plt.scatter(h,s)
axis.set_xlabel("Hue")
axis.set_ylabel("Saturation")
axis.set_zlabel("Value")
plt.show()

#axis.set_xlabel("Saturation")
#axis.set_ylabel("Value")
#plt.scatter(s,v)
#axis.scatter(s, v, facecolors=(0,0,0), marker=".")
#axis.set_xlabel("Hue")
#plt.show()




mask = cv2.inRange(hsv_nemo, light_orange, dark_orange)

result = cv2.bitwise_and(image, image, mask=mask)

plt.subplot(1, 2, 1)
plt.imshow(mask, cmap="gray")
plt.subplot(1, 2, 2)
plt.imshow(result)
plt.show()

"""
tomato = cv2.imread('test00001.png')

plt.imshow(cv2.cvtColor(tomato, cv2.COLOR_BGR2RGB))
"""
