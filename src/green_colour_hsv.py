# ===================================================================

# Task 2 : identify an image region by hue (e.g. green)

# Contact : amir.atapour-abarghouei@durham.ac.uk
# https://github.com/atapour/

# based on:
# https://github.com/atapour/chroma-keying/blob/main/src/hsv_colour.py

# License : MIT - https://opensource.org/licenses/MIT

# ===================================================================

import cv2
import numpy as np

# ===================================================================

# define the range of hues to detect - e.g. green

lower_green = np.array([55, 50, 50])
upper_green = np.array([95, 255, 255])

# ===================================================================

# define video capture with access to camera 0

camera = cv2.VideoCapture(0)

# define display window

window_name = "Live Camera Input with Green Hue Region"
cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

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

    mask = cv2.inRange(image_hsv, lower_green, upper_green)
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

    # overlay label

    cv2.putText(frame, 'Green Hue Isolated', (10, frame.shape[0] - 15),
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
