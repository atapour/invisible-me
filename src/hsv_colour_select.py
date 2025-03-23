# ===================================================================

# Task 2 : identify an image region by hue via point and click

# Contact : amir.atapour-abarghouei@durham.ac.uk
# https://github.com/atapour/

# based on:
# https://github.com/atapour/chroma-keying/blob/main/src/hsv_colour.py

# License : MIT - https://opensource.org/licenses/MIT

# ===================================================================

import cv2
import numpy as np

# ===================================================================

# define the range of hues to detect - set automatically using mouse

lower_bound = np.array([0, 0, 0])
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

cv2.VideoCapture(0,
cv2.CAP_V4L)

# define display window

window_name = "Live Camera Input with Selected Hue Region"
cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

# set the mouse call back function that will be called every time
# the mouse is clicked inside the display window

cv2.setMouseCallback(window_name, mouse_callback)

# ===================================================================

keep_processing = True

while (keep_processing):

    # read an image from the camera

    _, image = camera.read()

    # start a timer (to see how long processing and display takes)

    start_t = cv2.getTickCount()

    # convert the RGB images to HSV

    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # create a mask that identifies the pixels in the range of hues

    mask = cv2.inRange(image_hsv, lower_bound, upper_bound)
    mask_inverted = cv2.bitwise_not(mask)

    # create a grey image and black out the masked area

    image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_grey = cv2.bitwise_and(image_grey, image_grey, mask=mask_inverted)

    # black out unmasked area of original image

    image_masked = cv2.bitwise_and(image, image, mask=mask)

    # combine the two images for display

    image_grey = cv2.cvtColor(image_grey, cv2.COLOR_GRAY2BGR)
    frame = cv2.add(image_grey, image_masked)

    # stop the timer and convert to milliseconds
    # (to see how long processing and display takes)

    stop_t = ((cv2.getTickCount() - start_t) /
              cv2.getTickFrequency()) * 1000

    label = ('Processing time: %.2f ms' % stop_t) + \
        (' (Max Frames per Second (fps): %.2f' % (1000 / stop_t)) + ')'
    cv2.putText(frame, label, (0, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

    # overlay labels

    label1 = "left-click to select region by colour"
    label2 = "right-click to reset selected range"
    label3 = f'current range: {lower_bound} - {upper_bound}'

    cv2.putText(frame, label1, (10, frame.shape[0] - 85),
                cv2.FONT_HERSHEY_COMPLEX, 1, (123, 49, 126), 4)

    cv2.putText(frame, label2, (10, frame.shape[0] - 50),
                cv2.FONT_HERSHEY_COMPLEX, 1, (123, 49, 126), 4)

    cv2.putText(frame, label3, (10, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_COMPLEX, 1, (123, 49, 126), 4)

    # display image

    cv2.imshow(window_name, frame)

    # start the event loop - if user presses "q" then exit
    # wait 40ms for a key press from the user (i.e. 1000ms / 25 fps = 40 ms)

    key = cv2.waitKey(40) & 0xFF

    if (key == ord('q')):
        keep_processing = False

# ===================================================================

# Author : Amir Atapour-Abarghouei / Toby Breckon / Magnus Bordewich
# Copyright (c) 2023 Dept Computer Science, Durham University, UK

# ===================================================================
