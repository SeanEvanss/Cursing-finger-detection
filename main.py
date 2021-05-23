import time
import math
from cv2 import cv2
import mediapipe as mp
import numpy as np

def absol_distance(x1,y1,x2,y2):
    return int(math.sqrt((x1-x2)**2+(y1-y2)**2))

def finger_spread(finger_loc):

    return (absol_distance(finger_loc[15][0],finger_loc[15][1],finger_loc[19][0],finger_loc[19][1])<=
            absol_distance(finger_loc[9][0],finger_loc[9][1],finger_loc[17][0],finger_loc[17][1]))
def fist_clench(finger_loc):
    return (finger_spread(finger_loc) &
            #index finger
            (absol_distance(finger_loc[8][0],finger_loc[8][1],finger_loc[5][0],finger_loc[5][1])<=
            absol_distance(finger_loc[9][0],finger_loc[9][1],finger_loc[11][0],finger_loc[11][1])) &
            #ring finger
            (absol_distance(finger_loc[16][0],finger_loc[16][1],finger_loc[13][0],finger_loc[13][1])<=
            absol_distance(finger_loc[9][0],finger_loc[9][1],finger_loc[11][0],finger_loc[11][1])) &
            #pinkie finger
            (absol_distance(finger_loc[20][0],finger_loc[20][1],finger_loc[17][0],finger_loc[17][1])<=
            absol_distance(finger_loc[9][0],finger_loc[9][1],finger_loc[11][0],finger_loc[11][1])) )


def middle_finger_extended(finger_loc):

    return (absol_distance(finger_loc[12][0],finger_loc[12][1],finger_loc[9][0],finger_loc[9][1])>=
            absol_distance(finger_loc[5][0],finger_loc[5][1],finger_loc[17][0],finger_loc[17][1]))

def is_rude(finger_loc):

    return middle_finger_extended(finger_loc) & fist_clench(finger_loc)

capture= cv2.VideoCapture(0)
mpHands= mp.solutions.hands

hands= mpHands.Hands(max_num_hands=1, min_tracking_confidence=0.7, min_detection_confidence=0.7)
mpDraw= mp.solutions.drawing_utils

prev_time=0
curr_time=0

middle_finger=[10,11,12]
finger_loc=np.zeros((21,2))

while True:
    success, img = capture.read()

    imgRGB= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results= hands.process(imgRGB)

    if results.multi_hand_landmarks: #Checks if any hands are detected
        for indivHand in results.multi_hand_landmarks:
            for id, loc in enumerate(indivHand.landmark):

                h, w ,c =img.shape
                cx, cy= int(loc.x*w), int(loc.y*h)
                finger_loc[id][0] = cx
                finger_loc[id][1] = cy

                if (id in middle_finger):

                    cv2.circle(img, (cx,cy),30,(0,0,255))

                print(is_rude(finger_loc))
                if(is_rude(finger_loc)):
                    cv2.putText(img, 'RUDE', (200, 350), cv2.FONT_HERSHEY_PLAIN, 10, (0, 0, 255), 5)

            mpDraw.draw_landmarks(img, indivHand, mpHands.HAND_CONNECTIONS) #draw the landsmarks in each detected hands on screen

    curr_time=time.time()
    fps= 1/(curr_time-prev_time)
    prev_time=curr_time


    cv2.putText(img,str(int(fps)),(10,50), cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)

    key=cv2.waitKey(1)
    if key%256==27:
        break

    cv2.imshow("Displayed image", img)

