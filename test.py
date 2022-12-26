import cv2
import numpy as np
import matplotlib.pyplot as plt
<<<<<<< HEAD

def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    canny = cv2.Canny(blur, 50, 150)

    return canny

cap = cv2.VideoCapture('Easy_Test.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    
    canny_img = canny(frame)
    plt.imshow(canny_img)
    plt.show()
=======
import matplotlib.image as mpimg

cap = cv2.VideoCapture('lane_juniors.mp4')
while cap.isOpened():
    ret, frame = cap.read()
    # img = mpimg.imread(cap)
    plt.imshow(frame)
    plt.show()

# img = mpimg.imread('test_images/lane_juniors_frame122.jpg')

# plt.imshow(img)
# plt.show()


# from PerspectiveTransformation import *
# birdeye = PerspectiveTransformation()

# img = mpimg.imread('test_images/lane_juniors_frame122.jpg')
# img1 = birdeye.forward(img)
# plt.imshow(img1)
# plt.show()
>>>>>>> 69e88ff (tested new stuff)
