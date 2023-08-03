# Importing all necessary libraries
import cv2
import os

# Read the video from specified path
path = "test_vid2.mp4"
cam = cv2.VideoCapture("input_videos/" + path)
dir = path.strip('.mp4') + '_frames'

try:
    
    # creating a folder named data
    if not os.path.exists(dir):
        os.makedirs(dir)

# if not created then raise error
except OSError:
    print ('Error: Creating directory of data')

# frame
currentframe = 0

while(True):
    
    # reading from frame
    ret,frame = cam.read()

    if ret:
        # if video is still left continue creating images
        if currentframe%100 == 0:
            name = "./" + dir + "/" + path.strip('.mp4') + "_" + str(currentframe) + '.jpg'
            print ('Creating...' + name)
            # writing the extracted images
            cv2.imwrite(name, frame)

        # increasing counter so that it will
        # show how many frames are created
        currentframe += 1
    else:
        break

# Release all space and windows once done
cam.release()
cv2.destroyAllWindows()
