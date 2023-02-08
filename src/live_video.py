# ===================================================================

# Task 1 : capture live video from an attached camera

# Contact : amir.atapour-abarghouei@durham.ac.uk
# https://github.com/atapour/

# based on:
# https://github.com/atapour/chroma-keying/blob/main/src/live_video.py

# License : MIT - https://opensource.org/licenses/MIT

# ===================================================================

import cv2
import math

# ===================================================================

# this function is called as a call-back every time the trackbar is moved
# (here we just do nothing)

def nothing(x):
    pass


# ===================================================================

# define video capture with access to camera 0

camera = cv2.VideoCapture(0)

# define display window

window_name = "Live Camera Input"
cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

# add some track bar GUI controllers for smoothing filter parameters

neighbourhood_x = 3
cv2.createTrackbar(
    "filter size - X",
    window_name,
    neighbourhood_x,
    250,
    nothing)

neighbourhood_y = 3
cv2.createTrackbar(
    "filter size - Y",
    window_name,
    neighbourhood_y,
    250,
    nothing)

# ===================================================================

keep_processing = True

while (keep_processing):

    # read an image from the camera

    _, image = camera.read()

    # start a timer (to see how long processing and display takes)

    start_t = cv2.getTickCount()

    # perform any processing on the image here
    # by uncommenting (remove #) the following line, we can flip the image

    # cv2.flip(image, -1)

    # get parameters from track bars

    neighbourhood_x = cv2.getTrackbarPos("filter size - X", window_name)
    neighbourhood_y = cv2.getTrackbarPos("filter size - Y", window_name)

    # check filter sizes are greater than 3 and odd

    neighbourhood_x = max(3, neighbourhood_x)
    if not (neighbourhood_x % 2):
        neighbourhood_x = neighbourhood_x + 1

    neighbourhood_y = max(3, neighbourhood_y)
    if not (neighbourhood_y % 2):
        neighbourhood_y = neighbourhood_y + 1

    # performing smoothing on the image using a smoothing filter

    frame = cv2.GaussianBlur(image, (neighbourhood_x, neighbourhood_y), 0)

    # stop the timer and convert to milliseconds
    # (to see how long processing and display takes)

    stop_t = ((cv2.getTickCount() - start_t) /
              cv2.getTickFrequency()) * 1000

    label = ('Processing time: %.2f ms' % stop_t) + \
        (' (Max Frames per Second (fps): %.2f' % (1000 / stop_t)) + ')'
    cv2.putText(frame, label, (0, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

    # overlay label

    cv2.putText(
        frame,
        f'Smoothed Image {neighbourhood_x}x{neighbourhood_y}',
        (10, frame.shape[0]-15),
        cv2.FONT_HERSHEY_COMPLEX,
        1, (123, 49, 126), 5
        )

    # display image

    cv2.imshow(window_name, frame)

    # start the event loop - if user presses "q" then exit

    # wait 40ms or less for a key press from the user
    # depending on processing time taken (i.e. 1000ms / 25 fps = 40 ms)

    key = cv2.waitKey(max(2, 40 - int(math.ceil(stop_t)))) & 0xFF

    if (key == ord('q')):
        keep_processing = False

# ===================================================================

# Author : Amir Atapour-Abarghouei / Toby Breckon
# Copyright (c) 2023 Dept Computer Science, Durham University, UK

# ===================================================================
