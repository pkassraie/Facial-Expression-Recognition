import numpy as np
import cv2
import os
import glob
from ExtractNormalLandmarks import ExtractNormalLandMark

# accessing all the image and text folders
pathIm = 'facial_expression_dataset/cohn-kanade-images/'
pathEm = 'facial_expression_dataset/Emotion/'

peopleIm = [d for d in os.listdir(pathIm) if os.path.isdir(os.path.join(pathIm, d))]
peopleEm = [d for d in os.listdir(pathEm) if os.path.isdir(os.path.join(pathEm, d))]


peopleIm = [ pathIm + x for x in peopleIm]
peopleEm = [ pathEm + x for x in peopleEm]

emotionpics = list()
emotiontxts = list()

FinalFeatures = np.zeros((593,137),np.float64)
FinalLabels = np.zeros((593,1),np.float64)

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

    path =  glob.glob(os.path.join(i,'*.png'))[0]
    I = cv2.imread(path)
    Lands1,bias1 = ExtractNormalLandMark(I)

    path =  glob.glob(os.path.join(i,'*.png'))[-1]
    I = cv2.imread(path)
    Lands2,bias2 = ExtractNormalLandMark(I)

    Delta = np.zeros((2, 68), np.float64)
    Delta[0,:] = Lands2[0,:]-Lands1[0,:]
    Delta[1,:] = Lands2[1,:]-Lands1[1,:]


    FinalFeatures[n,0:68] = Delta[0,:]
    FinalFeatures[n,69:137] = Delta[1,:]
    print n
    n=n+1

np.savetxt("Features.csv",FinalFeatures,delimiter=',')
