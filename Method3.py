import numpy as np
import cv2
import os
import glob
from ExtractLargeLandmarks import ExtractLargeLandMarks



# accessing all the image and text folders
pathIm = 'facial_expression_dataset/cohn-kanade-images/'
pathEm = 'facial_expression_dataset/Emotion/'

peopleIm = [d for d in os.listdir(pathIm) if os.path.isdir(os.path.join(pathIm, d))]
peopleEm = [d for d in os.listdir(pathEm) if os.path.isdir(os.path.join(pathEm, d))]


peopleIm = [ pathIm + x for x in peopleIm]
peopleEm = [ pathEm + x for x in peopleEm]

emotionpics = list()
emotiontxts = list()

FinalFeatures = np.zeros((350,4557),np.float64)
FinalLabels = np.zeros((350,1),np.float64)

for i in peopleIm:
    temp = [d for d in os.listdir(i) if os.path.isdir(os.path.join(i, d))]
    temp = [ i + '/' + x for x in temp]
    emotionpics.append(temp)

for i in peopleEm:
    temp = [d for d in os.listdir(i) if os.path.isdir(os.path.join(i, d))]
    temp = [ i + '/' + x for x in temp]
    emotiontxts.append(temp)


# adding Labels
n=0
paths = []
for i in range(0,len(emotiontxts)):
    for j in range(0,len(emotiontxts[i])):
        Mainpath = emotiontxts[i][j] +'/'
        path = glob.glob(os.path.join(Mainpath,'*'))
        if (len(path)!=0):
            path = path[0]
            txt = open(path)
            lines = txt.readlines()[0]
            FinalLabels[n,0] = int(lines[3])
            paths.append(Mainpath.replace("Emotion","cohn-kanade-images"))
            n=n+1
np.savetxt("Labels.csv",FinalLabels,delimiter=',')


# calculating features for the images
n=0
for i in paths:
    path =  glob.glob(os.path.join(i,'*.png'))[-1]
    I = cv2.imread(path)
    Lands = ExtractLargeLandMarks(I)
    FinalFeatures[n,0:2278] = Lands[0,:]
    FinalFeatures[n,2279:4557] = Lands[1,:]
    print n
    n=n+1

np.savetxt("Features.csv",FinalFeatures,delimiter=',')
