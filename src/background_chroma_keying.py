# ===================================================================

# Task 4 : run a live chroma keying demo using a background image saved
#          as background.jpg

# Contact : amir.atapour-abarghouei@durham.ac.uk
# https://github.com/atapour/

# based on:
# https://github.com/tobybreckon/chroma-keying

# License : MIT - https://opensource.org/licenses/MIT

# ===================================================================

import cv2
import numpy as np
import urllib.request

# ===================================================================

# define the image to be used as the background
# this example is set up to only work with images from a url
# change the image by uncommenting these or adding your own

image_url = ('https://raw.githubusercontent.com/atapour/harry-potter-demo/'
             'main/img/background-hogwarts.jpg')

# image_url = ('https://www.thisisdurham.com/dbimgs/'
#              'durham-cathedral-background.jpg')

# image_url = ('https://static.wikia.nocookie.net/gameofthrones/'
#              'images/a/a9/Great_hall1x04.jpg/revision/latest?cb=20160717004721')

# image_url = ('https://upload.wikimedia.org/wikipedia/'
#              'commons/a/ac/Living_with_nature.jpg')

# ===================================================================

# define the range of hues to detect - set automatically using mouse

lower_bound = np.array([255, 0, 0])
upper_bound = np.array([255, 255, 255])

# ===================================================================

# mouse callback function - activated on any mouse event (click, movement)
# displays and sets Hue range based on right click location


def mouse_callback(event, x, y, flags, param):

    global upper_bound
    global lower_bound

    # records mouse events at position (x,y) in the image window
    # left button click prints colour HSV information and sets range

    if event == cv2.EVENT_LBUTTONDOWN:
        print("HSV colour @ position (%d,%d) = %s (bounds set with +/- 20)" %
              (x, y, ', '.join(str(i) for i in image_hsv[y, x])))

        # set Hue bounds on the Hue with +/- 15 threshold on the range

        upper_bound[0] = image_hsv[y, x][0] + 20
        lower_bound[0] = image_hsv[y, x][0] - 20

        # set Saturation and Value to eliminate very dark, noisy image regions

        lower_bound[1] = 50
        lower_bound[2] = 50

    # right button click resets HSV range

    elif event == cv2.EVENT_RBUTTONDOWN:

        lower_bound = np.array([0, 0, 0])
        upper_bound = np.array([255, 255, 255])


# ===================================================================

# define video capture with access to camera 0

camera = cv2.VideoCapture(0)

# define display window

window_name = "Live Camera Input with Chroma Keying Background"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# set the mouse call back function that will be called every time
# the mouse is clicked inside the display window

cv2.setMouseCallback(window_name, mouse_callback)

# ===================================================================

# first, read an image of the new background from the url
# and resize it to be the same size as our camera image

# we load the image directly from the web for simplicity

req = urllib.request.urlopen(image_url)
image_arr = np.asarray(bytearray(req.read()), dtype=np.uint8)

background = cv2.imdecode(image_arr, cv2.IMREAD_COLOR)

if background is None:
    print("\nbackground image file not successfully loaded!")
    exit(0)
else:
    _, image = camera.read()
    height, width, _ = image.shape
    background = cv2.resize(background, (width, height))

# ===================================================================

keep_processing = True

while (keep_processing):

    # read an image from the camera

    _, image = camera.read()

    # start a timer (to see how long processing and display takes)

    start_t = cv2.getTickCount()

    # convert the RGB images to HSV

    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # create a background mask that identifies the pixels in the range of hues

    background_mask = cv2.inRange(image_hsv, lower_bound, upper_bound)

    # logically invert the background mask to get foreground via logical NOT

    foreground_mask = cv2.bitwise_not(background_mask)

    # extract the set of contours around the foreground mask and then the
    # largest contour as the foreground object of interest. Update the
    # foreground mask with all the pixels inside this contour

    contours, _ = cv2.findContours(foreground_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    if (len(contours) > 0):
        largest_contour = max(contours, key=cv2.contourArea)
        foreground_mask_object = np.zeros((height, width))
        cv2.fillPoly(foreground_mask_object, [largest_contour], (255))

    # recompute the background mask based on the updated foreground mask

    background_mask = ((np.ones((height, width)) * 255)
                       - foreground_mask_object).astype('uint8')

    # construct 3-channel RGB feathered masks for blending

    foreground_mask_feathered = cv2.blur(foreground_mask, (15, 15)) / 255.0
    background_mask_feathered = cv2.blur(background_mask, (15, 15)) / 255.0
    background_mask_feathered = cv2.merge([background_mask_feathered,
                                           background_mask_feathered,
                                           background_mask_feathered])
    foreground_mask_feathered = cv2.merge([foreground_mask_feathered,
                                           foreground_mask_feathered,
                                           foreground_mask_feathered])

    # combine current camera image with new background via feathered blending

    choma_key_image = ((background_mask_feathered * background)
                       + (foreground_mask_feathered * image)).astype('uint8')

    # stop the timer and convert to milliseconds
    # (to see how long processing and display takes)

    stop_t = ((cv2.getTickCount() - start_t) /
              cv2.getTickFrequency()) * 1000

    label = ('Processing time: %.2f ms' % stop_t) + \
        (' (Max Frames per Second (fps): %.2f' % (1000 / stop_t)) + ')'
    cv2.putText(choma_key_image, label, (0, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

    # overlay labels

    label1 = "cover your background with the green fabric"
    label2 = "click on the (green) chroma keying material"
    label3 = "press space to re-capture background"

    cv2.putText(choma_key_image, label1, (10, choma_key_image.shape[0] - 85),
                cv2.FONT_HERSHEY_COMPLEX, 1, (123, 49, 126), 3)

    cv2.putText(choma_key_image, label2, (10, choma_key_image.shape[0] - 50),
                cv2.FONT_HERSHEY_COMPLEX, 1, (123, 49, 126), 3)

    cv2.putText(choma_key_image, label3, (10, choma_key_image.shape[0] - 15),
                cv2.FONT_HERSHEY_COMPLEX, 1, (123, 49, 126), 3)

    # display image with cloaking present

    cv2.imshow(window_name, choma_key_image)

    # start the event loop - if user presses "q" then exit
    # wait 40ms for a key press from the user (i.e. 1000ms / 25 fps = 40 ms)

    key = cv2.waitKey(40) & 0xFF

    # - if user presses q then exit

    if (key == ord('q')):
        keep_processing = False

# ===================================================================

# Author : Amir Atapour-Abarghouei / Toby Breckon
# Copyright (c) 2023 Dept Computer Science, Durham University, UK

# ===================================================================
