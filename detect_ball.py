#!/usr/bin/env python
import numpy as np
import cv2
import time

# Initial cv2 camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

ball_size = 0.075  # cm (diameter)
ball_objp = np.array([[ball_size/2,0,0],
                  [0,ball_size/2,0],
                  [ball_size,ball_size/2,0],
                  [ball_size/2,ball_size,0]], np.float32)

# intrinsic matrix from calibration
mtx = np.array([[653.32502, 0., 323.26114],
                [0. , 657.75146, 231.27424],
                [0. , 0. , 1.]])
dist = np.array([0.052520, -0.105545, 0.002344, -0.004994, 0.000000])
print("Starting with parameters: ")
print(mtx)
print(dist)


enable_offset = True
#===================== OFFSET ====================#
if(enable_offset):
    head_down_angle = 15
    x_offset = 0.40
    y_offset = -1.05
    z_offset = 0.58
    s = np.sin(np.radians(-head_down_angle-90))
    c = np.cos(np.radians(-head_down_angle-90))
    roll = np.array([[1, 0, 0, 0],
                        [0, c, -s, 0],
                        [0, s, c, 0],
                        [0, 0, 0, 1]])
    tran = np.array([[1, 0, 0, x_offset],
                        [0, 1, 0, y_offset],
                        [0, 0, 1, z_offset],
                        [0, 0, 0, 1]])

    world_mtx = tran@roll
    print(world_mtx)

    def cam_to_world(vec):
        vec = np.concatenate((vec,[[1]]))
        return world_mtx@vec
#=================================================#


new_frame_time = 0
prev_frame_time = 0
while True:
    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time

    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # preprocessing image
    # convert to hsv colorspace
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # lower bound and upper bound color
    lower_bound = np.array([150, 100, 20])   
    upper_bound = np.array([170, 255, 255])

    # find the colors within the boundaries
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    cv2.imshow('mask', mask)
    
    # find contours in the mask and initialize the current
    # (x, y) center of the ball
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)

    cnts = cnts[0]
    center = None
    # only proceed if at least one contour was found
    if(len(cnts) > 0):
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        # only proceed if the radius meets a minimum size
        #if radius > 20:
        ball_imgp = np.array([[x,y-radius],
            [x-radius,y],
            [x+radius,y],
            [x,y+radius]], np.float32)

        # Solve pnp
        ret, rvec, tvec = cv2.solvePnP(ball_objp, ball_imgp, mtx, dist)

        if(enable_offset):
            tvec = cam_to_world(tvec)

        # draw the circle and centroid on the frame,
        # then update the list of tracked points
        cv2.putText(frame, str(np.round(tvec[:3],2)), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, (0,255,0), 2, cv2.LINE_AA)
        cv2.circle(frame, (int(x), int(y)), int(radius),
            (0, 255, 255), 2)
        cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
            
    
    cv2.putText(frame, "fps: %.2f"%fps, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (0,255,0), 2, cv2.LINE_AA)
    cv2.imshow('frame', frame)
    if(cv2.waitKey(1) == ord('q')):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
