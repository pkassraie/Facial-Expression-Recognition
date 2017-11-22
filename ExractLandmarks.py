import numpy as np
import math
import dlib

def ExtractLandMark(image):

    # Initialization
    landx = np.zeros((1,68),np.float64)
    landy = np.zeros((1,68),np.float64)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    detections = detector(image, 1)
    for k,d in enumerate(detections):
        shape = predictor(image, d)

    # Loading Landmark coordinates
    for i in range(0,68):
        landx[0,i] = float(shape.part(i).x)
        landy[0,i] = float(shape.part(i).y)

    # Calcualting New coordinates with respect to the central landmark (nose probabily)
    xmean = np.mean(landx)
    ymean = np.mean(landy)
    landx2 = landx-xmean
    landy2 = landy-ymean
    
    # calculating rotation bias (points 0,28 and 0,30 are right below and above the nose)
    if landx[0,27] == landx[0,29]:
        biasdegree = 0
    else:
        biasdegree = int(math.atan((landy[0,27]-landy[0,29])/(landx[0,27]-landx[0,29]))*180/math.pi)
        biasdegree = biasdegree+90 if biasdegree<0 else biasdegree-90

    finalLands = np.zeros((4,68),np.float64)
    finalLands[0,:] = landx2
    finalLands[1,:] = landy2

    for i in range(0,68):
        finalLands[2,i] = np.linalg.norm((landy2[0,i],landx2[0,i]))
        finalLands[3,i] = (math.atan(landy2[0,i]/landx2[0,i])*180/math.pi) - biasdegree
    return finalLands

#frame= cv2.imread('facial_expression_dataset/cohn-kanade-images/S005/001/S005_001_00000005.png')
#lands = ExtractLandMark(frame)
