import numpy as np
import cv2
from collections import deque


def fun_emp(x):
    print("")


b_indices = [deque(maxlen=1024)]
g_indices = [deque(maxlen=1024)]
r_indices = [deque(maxlen=1024)]
y_indices = [deque(maxlen=1024)]
p_indices = [deque(maxlen=1024)]

blue_index = 0
green_index = 0
red_index = 0
yellow_index = 0
purple_index = 0

kernel = np.ones((5, 5), np.uint8)

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (128, 0, 128)]
colorIndex = 3

cv2.namedWindow("Pen Tracker")
cv2.createTrackbar("High Hue", "Pen Tracker", 40, 179, fun_emp)
cv2.createTrackbar("High Saturation", "Pen Tracker", 255, 255, fun_emp)
cv2.createTrackbar("High Value", "Pen Tracker", 255, 255, fun_emp)
cv2.createTrackbar("Low Hue", "Pen Tracker", 20, 180, fun_emp)
cv2.createTrackbar("Low Saturation", "Pen Tracker", 100, 255, fun_emp)
cv2.createTrackbar("Low Value", "Pen Tracker", 100, 255, fun_emp)
cv2.namedWindow('Pen Tracker', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Pen Tracker', 640, 480)

b_indices = [deque(maxlen=1024)]
g_indices = [deque(maxlen=1024)]
r_indices = [deque(maxlen=1024)]
y_indices = [deque(maxlen=1024)]
p_indices = [deque(maxlen=1024)]

blue_index = 0
green_index = 0
red_index = 0
yellow_index = 0
purple_index = 0

whiteboard = np.zeros((471, 636, 3)) + 255
whiteboard = cv2.rectangle(whiteboard, (40, 1), (140, 65), (0, 0, 0), -1)
whiteboard = cv2.rectangle(whiteboard, (40, 400), (140, 465), colors[0], -1)
whiteboard = cv2.rectangle(whiteboard, (160, 400), (255, 465), colors[1], -1)
whiteboard = cv2.rectangle(whiteboard, (275, 400), (370, 465), colors[2], -1)
whiteboard = cv2.rectangle(whiteboard, (390, 400), (485, 465), colors[3], -1)
whiteboard = cv2.rectangle(whiteboard, (505, 400), (600, 465), colors[4], -1)

cv2.putText(whiteboard, "BLUE", (49, 433), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(whiteboard, "ERASE ALL", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(whiteboard, "GREEN", (185, 433), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(whiteboard, "RED", (298, 433), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(whiteboard, "YELLOW", (420, 433), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(whiteboard, "PURPLE", (520, 433), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)

cap = cv2.VideoCapture(0)

while True:
    ret, vid = cap.read()
    vid = cv2.flip(vid, 1)
    hsv = cv2.cvtColor(vid, cv2.COLOR_BGR2HSV)

    u_hue = cv2.getTrackbarPos("High Hue", "Pen Tracker")
    u_saturation = cv2.getTrackbarPos("High Saturation", "Pen Tracker")
    u_value = cv2.getTrackbarPos("High Value", "Pen Tracker")
    l_hue = cv2.getTrackbarPos("Low Hue", "Pen Tracker")
    l_saturation = cv2.getTrackbarPos("Low Saturation", "Pen Tracker")
    l_value = cv2.getTrackbarPos("Low Value", "Pen Tracker")
    High_hsv = np.array([u_hue, u_saturation, u_value])
    Low_hsv = np.array([l_hue, l_saturation, l_value])

    vid = cv2.rectangle(vid, (40, 400), (140, 465), colors[0], -1)
    vid = cv2.rectangle(vid, (40, 1), (140, 65), (255, 255, 255), -1)
    vid = cv2.rectangle(vid, (160, 400), (255, 465), colors[1], -1)
    vid = cv2.rectangle(vid, (275, 400), (370, 465), colors[2], -1)
    vid = cv2.rectangle(vid, (390, 400), (485, 465), colors[3], -1)
    vid = cv2.rectangle(vid, (505, 400), (600, 465), colors[4], -1)

    cv2.putText(vid, "BLUE", (49, 433), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(vid, "ERASE ALL", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(vid, "GREEN", (185, 433), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(vid, "RED", (298, 433), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(vid, "YELLOW", (420, 433), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(vid, "PURPLE", (520, 433), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    Mask = cv2.inRange(hsv, Low_hsv, High_hsv)
    Mask = cv2.erode(Mask, kernel, iterations=1)
    Mask = cv2.morphologyEx(Mask, cv2.MORPH_OPEN, kernel)
    Mask = cv2.dilate(Mask, kernel, iterations=1)

    cnts, _ = cv2.findContours(Mask.copy(), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    center = None

    if len(cnts) > 0:

        cnt = sorted(cnts, key=cv2.contourArea, reverse=True)[0]

        ((x, y), radius) = cv2.minEnclosingCircle(cnt)

        cv2.circle(vid, (int(x), int(y)), int(radius), (0, 255, 0), 2)

        M = cv2.moments(cnt)
        center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))

        if center[1] >= 400 or (center[1] <= 65 and center[0] <= 140):
            if center[1] <= 65 and center[0] <= 140:
                b_indices = [deque(maxlen=512)]
                g_indices = [deque(maxlen=512)]
                r_indices = [deque(maxlen=512)]
                y_indices = [deque(maxlen=512)]

                blue_index = 0
                green_index = 0
                red_index = 0
                yellow_index = 0

                whiteboard[:, :, :] = 255

            elif 40 <= center[0] <= 140:
                colorIndex = 0
            elif 160 <= center[0] <= 255:
                colorIndex = 1  # Blue
            elif 275 <= center[0] <= 370:
                colorIndex = 2  # Green
            elif 390 <= center[0] <= 485:
                colorIndex = 3  # Red
            elif 505 <= center[0] <= 600:
                colorIndex = 4  # Yellow
        else:
            if colorIndex == 0:
                b_indices[blue_index].appendleft(center)
            elif colorIndex == 1:
                g_indices[green_index].appendleft(center)
            elif colorIndex == 2:
                r_indices[red_index].appendleft(center)
            elif colorIndex == 3:
                y_indices[yellow_index].appendleft(center)
            elif colorIndex == 4:
                p_indices[purple_index].appendleft(center)
    else:
        b_indices.append(deque(maxlen=512))
        blue_index += 1
        g_indices.append(deque(maxlen=512))
        green_index += 1
        r_indices.append(deque(maxlen=512))
        red_index += 1
        y_indices.append(deque(maxlen=512))
        yellow_index += 1
        p_indices.append(deque(maxlen=512))
        purple_index += 1

    points = [b_indices, g_indices, r_indices, y_indices, p_indices]

    for i in range(len(points)):
        for j in range(len(points[i])):
            for k in range(1, len(points[i][j])):
                if points[i][j][k - 1] is None or points[i][j][k] is None:
                    continue
                cv2.line(vid, points[i][j][k - 1], points[i][j][k], colors[i], 2)
                cv2.line(whiteboard, points[i][j][k - 1], points[i][j][k], colors[i], 2)

    cv2.imshow("Cam", vid)
    if cv2.waitKey(1) & 0xFF == ord("w"):
        cv2.imshow("Whiteboard", whiteboard)
    cv2.imshow("mask", Mask)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
