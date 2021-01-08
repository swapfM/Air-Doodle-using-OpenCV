import numpy
from cv2 import cv2
from collections import deque
import pickle


# Getting already saved values from save-file
try:
    HSVRangeFileRead = open('HSV_Range.pckl', 'rb')
except:
    HSVRangeFileWrite = open('HSV_Range.pckl', 'wb')
    pickle.dump([[179, 255, 255], [0, 0, 0]], HSVRangeFileWrite)
    HSVRangeFileWrite.close()
finally:
    HSVRangeFileRead = open('HSV_Range.pckl', 'rb')
    savedValues = pickle.load(HSVRangeFileRead)
    HSVRangeFileRead.close()


# Setting global utility variables
frameHeight = 480
frameWidth = 640
gettingHSVFrom = [int(frameWidth/2), int(frameHeight/2)]
settingHSVRange = False
minHSV = numpy.array(savedValues[0])
maxHSV = numpy.array(savedValues[1])
averageHSV = numpy.array([90, 128, 128])
totalSettingCaptures = 0
recentHSV = deque(maxlen=8)
kernel = numpy.ones((5, 5), numpy.uint8)
videoCaptured = cv2.VideoCapture(0)
capturedSuccessfully, frame = videoCaptured.read()
if capturedSuccessfully:
    frameHeight = numpy.size(frame, 0)
    frameWidth = numpy.size(frame, 1)
    gettingHSVFrom = [int(frameWidth/2), int(frameHeight/2)]
    print("Height Width = ", frameHeight, frameWidth)
videoCapturedWindow = cv2.namedWindow('Video', flags=cv2.WINDOW_AUTOSIZE)


# Function to be called on mouse activity in the 'Video' Window
def onMouse(event, x, y, flags, param):
    global settingHSVRange
    global gettingHSVFrom
    if event == cv2.EVENT_MOUSEMOVE and settingHSVRange:
        gettingHSVFrom = [min(x, frameWidth-1), min(y, frameHeight-1)]
    elif event == cv2.EVENT_FLAG_LBUTTON:
        settingHSVRange = True
        gettingHSVFrom = [min(x, frameWidth-1), min(y, frameHeight-1)]
    elif event == cv2.EVENT_LBUTTONUP:
        settingHSVRange = False
cv2.setMouseCallback('Video', onMouse)


# Infinite loop to display live video
while True:
    notification = ''
    key = cv2.waitKey(1) & 0xFF
    if key == ord('r') or key == ord('R'):                  # If 'R' key is pressed
        minHSV = numpy.array([179, 255, 255])
        maxHSV = numpy.array([0, 0, 0])
        averageHSV = numpy.array([90, 128, 128])
        recentHSV.clear()
        totalSettingCaptures = 0
        settingHSVRange = False
    elif key == ord(chr(13)):                               # If Enter key is pressed
        HSVRangeFileWrite = open('HSV_Range.pckl', 'wb')
        pickle.dump([minHSV, maxHSV], HSVRangeFileWrite)
        HSVRangeFileWrite.close()
        print("MinHSV = " + str(minHSV))
        print("MaxHSV = " + str(maxHSV))
        notification = 'SAVED'
    elif key == ord(chr(27)):                               # If Escape key is pressed
        break

    capturedSuccessfully, frame = videoCaptured.read()
    if capturedSuccessfully:

        frame = cv2.flip(frame, 1)
        frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        targetPoint = frameHSV[gettingHSVFrom[1], gettingHSVFrom[0]]

        if settingHSVRange:

            # Using mean of recent captured values to minimize the effect of fauly clicks (if any)
            recentHSV.append(targetPoint)
            sumOfrecentHSV = numpy.array([0, 0, 0])
            averageOfrecentHSV = numpy.array([0, 0, 0])
            for i in range(0, 3):
                for hsv in recentHSV:
                    sumOfrecentHSV[i] += hsv[i]
                averageOfrecentHSV[i] = sumOfrecentHSV[i] / len(recentHSV)
            totalSettingCaptures += 1

            for i in range(0, 3):

                averageHSV[i] = (totalSettingCaptures - 1) * averageHSV[i] + targetPoint[i]
                averageHSV[i] /= totalSettingCaptures

                if averageOfrecentHSV[i] < minHSV[i]:
                    minHSV[i] = averageOfrecentHSV[i]
                if averageOfrecentHSV[i] > maxHSV[i]:
                    maxHSV[i] = averageOfrecentHSV[i]

        # Making the Final Frame image to display
        if notification != '':
            frame = numpy.zeros((frameHeight, frameWidth), numpy.uint8)
            cv2.putText(frame, notification, (20, 40), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 255), 2)
        else:
            maskFrame = cv2.inRange(frameHSV, minHSV, maxHSV)
            invertedMaskFrame = cv2.bitwise_not(maskFrame)
            negetiveMaskedFrame = cv2.bitwise_not(frame, mask=maskFrame)
            frameWithoutMaskedPart = cv2.bitwise_and(frame, frame, mask=invertedMaskFrame)
            frame = cv2.bitwise_or(negetiveMaskedFrame, frameWithoutMaskedPart)

        cv2.imshow('Video', frame)

cv2.destroyAllWindows()
videoCaptured.release()
