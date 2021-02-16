import numpy as np 
import pandas as pd 
import cv2
import matplotlib.pyplot as plt 


# image = cv2.imread('plane.jpg')

# blur_image = cv2.blur(image, (5,5))

# plt.imshow(blur_image)
# # plt.show()

# # blur with custom kernel
# kernel = np.ones([10,10])/100
# blur_kernel = cv2.filter2D(image, -1, kernel)
# plt.imshow(blur_kernel)
# # plt.show()

# # sharpening image with custom kernel for sharpen the image 

# kernel = np.array([
#     [0,-1,0],[-1,5,-1],[0,-1,0]
# ])

# sharp_image = cv2.filter2D(image, -1, kernel)
# plt.imshow(sharp_image)
# # plt.show()

# # enhancing contrast 
# image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
# # contrast_image_yuv = cv2.equalizeHist(image_yuv)
# # contrast_image = cv2..cvtColor(contrast_image_yuv, cv2.COLOR_YUV2BGR)
# # plt.imshow(contrast_image)
# # plt.show()
# print(kernel)

# # making mask for lower and upper blue 

# image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# lower_blue = [50,100,50]
# upper_blue = [130,255,255]

# mask = cv2.inRange(image_hsv, lower_blue,upper_blue)
# image_bgr_masked = cv2.bitwise_and(image_hsv,image_hsv,mask=mask)
# image_rgb_masked = cv2.cvtColor(image_bgr_masked, cv2.COLOR_BGR2RGB)
# # plt.imshow(image_rgb_masked)

# # plt.imshow(mask)
# # plt.show()
video = cv2.VideoCapture(0)

while(True):
    _,image = video.read()
    cv2.imshow('frame', image)

    if ~(cv2.waitKey(1):
        break

video.release()
cv2.destroyAllWindows()