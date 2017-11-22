import numpy as np
import dlib

def ExtractLargeLandMarks(image):
    # Initialization
    landx = np.zeros((1,68),np.float64)
    landy = np.zeros((1,68),np.float64)
    finalLands = np.zeros((2,2278),np.float64)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    detections = detector(image, 1)
    for k,d in enumerate(detections):
        shape = predictor(image, d)

    # Loading Landmark coordinates
    for i in range(0,68):
        landx[0,i] = float(shape.part(i).x)
        landy[0,i] = float(shape.part(i).y)

    n=0
    for i in range(0,landx.shape[1]):
        for j in range(i+1,landy.shape[1]):
            finalLands[0,n] = landx[0,j]-landx[0,i]
            finalLands[1,n] = landy[0,j]-landy[0,i]
            n=n+1

    return finalLands

