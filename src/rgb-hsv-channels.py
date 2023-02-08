# ===================================================================

# Task 2 : display live video from camera as RGB and HSV channels

# Contact : amir.atapour-abarghouei@durham.ac.uk
# https://github.com/atapour/

# License : MIT - https://opensource.org/licenses/MIT

# ===================================================================

import cv2
import numpy as np
import warnings

# ===================================================================

warnings.filterwarnings("ignore")
keep_processing = True

# ===================================================================

# define video capture with access to camera 0

camera = cv2.VideoCapture(0)

# define display window name

window_name = "Live Camera - RGB and HSV Colour Channels"  # window name

# ===================================================================

keep_processing = True

while (keep_processing):

    # read an image from the camera

    _, rgb = camera.read()

    # start a timer (to see how long processing and display takes)

    start_t = cv2.getTickCount()

    # *******************************

    # construct RGB channel view (N.B. OpenCV is BGR, not RGB channel ordering)

    red = np.zeros(rgb.shape, dtype=np.uint8)
    red[:, :, 2] = rgb[:, :, 2]

    green = np.zeros(rgb.shape, dtype=np.uint8)
    green[:, :, 1] = rgb[:, :, 1]

    blue = np.zeros(rgb.shape, dtype=np.uint8)
    blue[:, :, 0] = rgb[:, :, 0]

    # convert image to hsv

    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)

    # get HSV channels separately

    hue = np.zeros(hsv.shape, dtype=np.uint8)
    hue[:, :, 0] = np.uint8(hsv[:, :, 0] * (0.7))
    hue[:, :, 1] = np.ones(hsv[:, :, 1].shape) * 255
    hue[:, :, 2] = np.ones(hsv[:, :, 2].shape) * 255
    hue = cv2.cvtColor(hue, cv2.COLOR_HSV2RGB)  # RGB better

    saturation = cv2.cvtColor(hsv[:, :, 1], cv2.COLOR_GRAY2BGR)
    value = cv2.cvtColor(hsv[:, :, 2], cv2.COLOR_GRAY2BGR)

    # overlay corresponding labels on the images

    cv2.putText(rgb, 'Input Image', (10, rgb.shape[0] - 15),
            cv2.FONT_HERSHEY_COMPLEX, 2, (123, 49, 126), 6)
    cv2.putText(red, 'RGB - Red', (10, red.shape[0] - 15),
            cv2.FONT_HERSHEY_COMPLEX, 2, (123, 49, 126), 6)
    cv2.putText(green, 'RGB - Green', (10, green.shape[0] - 15),
            cv2.FONT_HERSHEY_COMPLEX, 2, (123, 49, 126), 6)
    cv2.putText(blue, 'RGB - Blue', (10, blue.shape[0] - 15),
            cv2.FONT_HERSHEY_COMPLEX, 2, (123, 49, 126), 6)
    cv2.putText(hue, 'HSV - Hue', (10, hue.shape[0] - 15),
            cv2.FONT_HERSHEY_COMPLEX, 2, (123, 49, 126), 6)
    cv2.putText(saturation, 'HSV - Saturation', (10, saturation.shape[0] - 15),
            cv2.FONT_HERSHEY_COMPLEX, 2, (123, 49, 126), 6)
    cv2.putText(value, 'HSV - Value', (10, value.shape[0] - 15),
            cv2.FONT_HERSHEY_COMPLEX, 2, (123, 49, 126), 6)

    # stack the images into a grid

    im_1 = cv2.hconcat([rgb, red, green, blue])
    im_2 = cv2.hconcat([rgb, hue, saturation, value])
    output = cv2.vconcat([im_1, im_2])

    # *******************************
        
    # stop the timer and convert to ms. (to see how long processing and
    # display takes)

    stop_t = ((cv2.getTickCount() - start_t) /
              cv2.getTickFrequency()) * 1000

    label = ('Processing time: %.2f ms' % stop_t) + \
        (' (Max Frames per Second (fps): %.2f' % (1000 / stop_t)) + ')'
    cv2.putText(output, label, (0, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

   # display image

    cv2.imshow(window_name, output)

    # wait 40ms for a key press from the user (i.e. 1000ms / 25 fps = 40 ms)

    key = cv2.waitKey(40) & 0xFF

    # It can also be set to detect specific key strokes by recording which
    # key is pressed

    # e.g. if user presses "q" then exit

    if (key == ord('q')):
        keep_processing = False

    # - if user presses "f" then switch to full screen

    elif (key == ord('f')):
        print("\n -- toggle fullscreen.")
        last_fs = cv2.getWindowProperty(window_name,
                                        cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,
                            cv2.WINDOW_FULLSCREEN &
                            ~(int(last_fs)))

# ===================================================================

# Author : Amir Atapour-Abarghouei / Toby Breckon
# Copyright (c) 2023 Dept Computer Science, Durham University, UK

# ===================================================================
