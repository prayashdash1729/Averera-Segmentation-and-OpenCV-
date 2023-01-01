import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# def canny(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#     blur = cv2.GaussianBlur(gray, (5,5), 0)
#     canny = cv2.Canny(blur, 50, 150)

#     return canny

# cap = cv2.VideoCapture('Easy_Test.mp4')

# while cap.isOpened():
#     ret, frame = cap.read()
    
#     canny_img = canny(frame)
#     plt.imshow(canny_img)
#     plt.show()
# import matplotlib.image as mpimg

# cap = cv2.VideoCapture('lane_juniors.mp4')
# while cap.isOpened():
#     ret, frame = cap.read()
#     # img = mpimg.imread(cap)
#     plt.imshow(frame)
#     plt.show()

img1 = mpimg.imread('D:\Prayash\Lane_detection\Easy_Test\Easy_test_frames\Easy_Test_frame480.jpg')
img2 = mpimg.imread('D:\Prayash\Lane_detection\Easy_Test\Easy_test_frames\Easy_Test_frame270.jpg')

f = plt.figure(figsize=(24, 16))
ax1 = f.add_subplot(1, 2, 1)
ax2 = f.add_subplot(1, 2, 2)
ax1.imshow(img1)
ax2.imshow(img2)
plt.show()

# from PerspectiveTransformation import *
# birdeye = PerspectiveTransformation()

# img = mpimg.imread('test_images/lane_juniors_frame122.jpg')
# # img1 = birdeye.forward(img)
# plt.imshow(img)
# plt.show()
