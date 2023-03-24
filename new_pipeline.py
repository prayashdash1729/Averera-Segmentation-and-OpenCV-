import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from docopt import docopt


class PerspectiveTransformation:
    """ This a class for transforming image between front view and top view

    Attributes:
        src (np.array): Coordinates of 4 source points
        dst (np.array): Coordinates of 4 destination points
        M (np.array): Matrix to transform image from front view to top view
        M_inv (np.array): Matrix to transform image from top view to front view
    """
    def __init__(self):
        """Init PerspectiveTransformation."""
        height, width = (1080, 1920)

        region_of_interest = np.array([
        [(400, 500), (0, 900), (width, 900), (1600, 500)]
        ])
        
        self.src = np.float32([(400, 500), 
                               (0, 900), 
                               (width, 900), 
                               (1600, 500)])
        self.dst = np.float32([(100, 0),
                               (100, height),
                               (width-100, height),
                               (width-100, 0)])

        self.M = cv2.getPerspectiveTransform(self.src, self.dst)
        self.M_inv = cv2.getPerspectiveTransform(self.dst, self.src)

    def forward(self, img, img_size=(1920, 1080), flags=cv2.INTER_LINEAR):
        """ Take a front view image and transform to top view

        Parameters:
            img (np.array): A front view image
            img_size (tuple): Size of the image (width, height)
            flags : flag to use in cv2.warpPerspective()

        Returns:
            Image (np.array): Top view image
        """
        return cv2.warpPerspective(img, self.M, img_size, flags=flags)

    def backward(self, img, img_size=(1920, 1080), flags=cv2.INTER_LINEAR):
        """ Take a top view image and transform it to front view

        Parameters:
            img (np.array): A top view image
            img_size (tuple): Size of the image (width, height)
            flags (int): flag to use in cv2.warpPerspective()

        Returns:
            Image (np.array): Front view image
        """
        return cv2.warpPerspective(img, self.M_inv, img_size, flags=flags)

def threshold_rel(img, lo, hi):
    vmin = np.min(img)
    vmax = np.max(img)
    
    vlo = vmin + (vmax - vmin) * lo
    vhi = vmin + (vmax - vmin) * hi
    return np.uint8((img >= vlo) & (img <= vhi)) * 255

def threshold_abs(img, lo, hi):
    return np.uint8((img >= lo) & (img <= hi)) * 255


class Thresholding:
    """ This class is for extracting relevant pixels in an image.
    """
    def __init__(self):
        """ Init Thresholding."""
        pass

    def forward(self, img):
        """ Take an image and extract all relavant pixels.

        Parameters:
            img (np.array): Input image

        Returns:
            binary (np.array): A binary image represent all positions of relavant pixels.
        """
        
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        h_channel = hls[:,:,0]
        l_channel = hls[:,:,1]
        s_channel = hls[:,:,2]
        v_channel = hsv[:,:,2]

        lanes = threshold_rel(v_channel, 0.83, 1.0)
        lanes[:,600:1300]=0
        return lanes
    

def hist(img):
    bottom_half = img[img.shape[0]//2:,:]
    return np.sum(bottom_half, axis=0)

class LaneLines:
    """ Class containing information about detected lane lines.

    Attributes:
        left_fit (np.array): Coefficients of a polynomial that fit left lane line
        right_fit (np.array): Coefficients of a polynomial that fit right lane line
        parameters (dict): Dictionary containing all parameters needed for the pipeline
        debug (boolean): Flag for debug/normal mode
    """
    def __init__(self):
        """Init Lanelines.

        Parameters:
            left_fit (np.array): Coefficients of polynomial that fit left lane
            right_fit (np.array): Coefficients of polynomial that fit right lane
            binary (np.array): binary image
        """
        self.img = None
        self.leftx_base = None
        self.rightx_base = None
        self.left_fit = None
        self.right_fit = None
        self.binary = None
        self.nonzero = None
        self.nonzerox = None
        self.nonzeroy = None
        self.clear_visibility = True
        self.dir = []
    
        # HYPERPARAMETERS
        # Number of sliding windows
        self.nwindows = 9
        # Width of the the windows +/- margin
        self.margin = 300
        # Mininum number of pixels found to recenter window
        self.minpix = 450

    def forward(self, img):
        """Take a image and detect lane lines.

        Parameters:
            img (np.array): An binary image containing relevant pixels

        Returns:
            Image (np.array): An RGB image containing lane lines pixels and other details
        """
        self.img = np.copy(img)
        self.extract_features(img)
        return self.fit_poly(img)

    def pixels_in_window(self, center, margin, height):
        """ Return all pixel that in a specific window

        ->  Now that we have set the left and right base, we slide our windows vertically 
            upward from the base points n times where n = self.nwindows and check for white 
            pixels in each window. The height of each window is img.shape[0]//self.nwindows 
            and the width is (self.margin)*2. We append leftx, lefty, rightx, righty appropriately. 
            If the number of white pixels in that window is more than self.minpix, then we 
            shift the center of the window to the mean of x coordinates of those white pixels.

        Parameters:
            center (tuple): coordinate of the center of the window
            margin (int): half width of the window
            height (int): height of the window

        Returns:
            pixelx (np.array): x coordinates of white pixels that lie inside the window
            pixely (np.array): y coordinates of white pixels that lie inside the window
        """
        topleft = (center[0]-margin, center[1]-height//2)
        bottomright = (center[0]+margin, center[1]+height//2)

        condx = (topleft[0] <= self.nonzerox) & (self.nonzerox <= bottomright[0])
        condy = (topleft[1] <= self.nonzeroy) & (self.nonzeroy <= bottomright[1])
        return self.nonzerox[condx&condy], self.nonzeroy[condx&condy]

    def extract_features(self, img):
        """ Extract features from a binary image

        Parameters:
            img (np.array): A binary image
        """
        self.img = img
        # Height of of windows - based on nwindows and image shape
        self.window_height = np.int32(img.shape[0]//self.nwindows)

        # Identify the x and y positions of all nonzero pixel in the image
        self.nonzero = img.nonzero()
        self.nonzerox = np.array(self.nonzero[1])
        self.nonzeroy = np.array(self.nonzero[0])

    def find_lane_pixels(self, img):
        """Find lane pixels from a binary warped image.

        Parameters:
            img (np.array): A binary warped image

        Returns:
            leftx (np.array): x coordinates of left lane pixels
            lefty (np.array): y coordinates of left lane pixels
            rightx (np.array): x coordinates of right lane pixels
            righty (np.array): y coordinates of right lane pixels
            out_img (np.array): A RGB image that use to display result later on.
        """
        assert(len(img.shape) == 2)

        # Create an output image to draw on and visualize the result
        out_img = np.dstack((img, img, img))

        # Checks vertically (axis0) for max amount of white pixels and sets that as base
        histogram = hist(img)
        midpoint = histogram.shape[0]//2
        self.leftx_base = np.argmax(histogram[:midpoint])
        self.rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Current position to be update later for each window in nwindows
        leftx_current = self.leftx_base
        rightx_current = self.rightx_base
        y_current = img.shape[0] + self.window_height//2

        # Create empty lists to recieve left and right lane pixel
        leftx, lefty, rightx, righty = [], [], [], []

        # Step through the windows one by one
        for _ in range(self.nwindows):
            y_current -= self.window_height
            center_left = (leftx_current, y_current)
            center_right = (rightx_current, y_current)

            good_left_x, good_left_y = self.pixels_in_window(center_left, self.margin, self.window_height)
            good_right_x, good_right_y = self.pixels_in_window(center_right, self.margin, self.window_height)

            # Append these indices to the lists
            leftx.extend(good_left_x)
            lefty.extend(good_left_y)
            rightx.extend(good_right_x)
            righty.extend(good_right_y)

            if len(good_left_x) > self.minpix:
                leftx_current = np.int32(np.mean(good_left_x))
            if len(good_right_x) > self.minpix:
                rightx_current = np.int32(np.mean(good_right_x))

        return leftx, lefty, rightx, righty, out_img

    def fit_poly(self, img):
        """Find the lane line from an image and draw it.

        Parameters:
            img (np.array): a binary warped image

        Returns:
            out_img (np.array): a RGB image that have lane line drawn on that.
        """

        leftx, lefty, rightx, righty, out_img = self.find_lane_pixels(img)

        if len(lefty) > 1500:
            self.left_fit = np.polyfit(lefty, leftx, 2)
        if len(righty) > 1500:
            self.right_fit = np.polyfit(righty, rightx, 2)

        # Generate x and y values for plotting
        maxy = img.shape[0] - 1
        miny = img.shape[0] // 5
        # miny = 0

        # if len(lefty):
        #     maxy = max(maxy, np.max(lefty))

        # if len(righty):
        #     maxy = max(maxy, np.max(righty))

        ploty = np.linspace(miny, maxy, 1000)


        # x' = a*y^2 + b*y + c
        if self.left_fit is not None and self.right_fit is not None:
            left_fitx = self.left_fit[0]*ploty**2 + self.left_fit[1]*ploty + self.left_fit[2]
            right_fitx = self.right_fit[0]*ploty**2 + self.right_fit[1]*ploty + self.right_fit[2]
            mid_fitx = (left_fitx + right_fitx)/2

            # Visualization
            for i, y in enumerate(ploty):
                l = int(left_fitx[i])
                r = int(right_fitx[i])
                y = int(y)
                cv2.line(out_img, (l, y), (r, y), (0, 255, 0))
                print(out_img.shape)

            # drawing the middle line of lane
            for i in range(len(ploty)-1):
                y1 = int(ploty[i])
                x1 = int(mid_fitx[i])
                y2 = int(ploty[i+1])
                x2 = int(mid_fitx[i+1])
                cv2.line(out_img, (x1, y1), (x2, y2), (0, 0, 255), 15)

            # drawing the center of the vehicle
            # cv2.line(out_img, (1920, 2160), (1920, 2000), (255,0,0), 10)

            return out_img
        
        else:
            return self.img
        
    def plot(self, out_img):
        np.set_printoptions(precision=6, suppress=True)
        # lR, rR, pos = self.measure_curvature()
        error = self.error()

        W = 600
        H = 50
        widget = np.copy(out_img[:H, :W])
        widget //= 2
        widget[0,:] = [0, 0, 255]
        widget[-1,:] = [0, 0, 255]
        widget[:,0] = [0, 0, 255]
        widget[:,-1] = [0, 0, 255]
        out_img[:H, :W] = widget
        
        cv2.putText(
            out_img,
            "Vehicle is {:.2f}m away from center".format(error),
            org=(10, 35),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(255, 255, 255),
            thickness=2)

        return out_img

    def error(self):
        # lane width = 3m
        xm = 3/(self.rightx_base - self.leftx_base)

        xl = np.copy(self.leftx_base)
        xr = np.copy(self.rightx_base)
        er = (1920//2 - (xl+xr)//2)*xm
        return er
        

class FindLaneLines:
    """ This class is for parameter tunning.

    Attributes:
        ...
    """
    def __init__(self):
        """ Init Application"""
        # self.calibration = CameraCalibration('camera_cal', 9, 6)
        self.thresholding = Thresholding()
        self.transform = PerspectiveTransformation()
        self.lanelines = LaneLines()

    def forward(self, img):
        out_img = np.copy(img)
        # img = self.calibration.undistort(img)
        img = self.transform.forward(img)
        img = self.thresholding.forward(img)
        img = self.lanelines.forward(img)
        img = self.transform.backward(img)

        out_img = cv2.addWeighted(out_img, 1, img, 0.6, 0)
        out_img = self.lanelines.plot(out_img)
        error = self.lanelines.error()
        return error, out_img

    def process_input_frame(self, input_frame):
        # img = mpimg.imread(input_path)
        error, out_img = self.forward(input_frame)
        # mpimg.imsave(output_path, out_img)
        return error, out_img


if __name__ == "__main__":

    video = 'input_videos/new_road3.mp4'
    cap = cv2.VideoCapture(video)
    
    app = FindLaneLines()

    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            count+=1

            if count%3 == 0:
                error, out_img = app.process_input_frame(frame)
                print(error)

                cv2.namedWindow("out", cv2.WINDOW_NORMAL)
                cv2.imshow("out", out_img)
                if cv2.waitKey(20) & 0xFF == ord('q'):
                    break
