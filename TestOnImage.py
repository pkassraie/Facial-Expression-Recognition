from sklearn.externals import joblib
from ExtractNormalLandmarks import ExtractNormalLandMark
from ExtractLargeLandmarks import ExtractLargeLandMarks
from ExractLandmarks import ExtractLandMark
import cv2
import numpy as np

Emotions = ["Angry", "Contempt","Disgust","Fear", "Happy", "Sadness","Surprised"]
img = cv2.imread('test2.png')
I = img.copy()
J = cv2.imread('test1.png')


TestX1 = np.zeros((1,137),np.float64)
Lands = ExtractLandMark(I)
TestX1[0,0:68] = Lands[2,:]
TestX1[0,69:137] = Lands[3,:]

TestX2 = np.zeros((1,137),np.float64)
Lands1,bias1 = ExtractNormalLandMark(J)
Lands2,bias2 = ExtractNormalLandMark(I)
TestX2[0,0:68] = Lands2[0,:]-Lands1[0,:]
TestX2[0,69:137] = Lands2[1,:]-Lands1[1,:]

TestX3 = np.zeros((1,4557),np.float64)
Lands = ExtractLargeLandMarks(I)
TestX3[0,0:2278] = Lands[0,:]
TestX3[0,2279:4557] = Lands[1,:]

clf1 = joblib.load('Classifier1/Clf_LinSVC.pkl')
clf2 = joblib.load('Classifier2/Clf_LinSVC.pkl')
clf3 = joblib.load('Classifier3/Clf_LinSVC.pkl')

Y_pred1 = clf1.predict(TestX1)
print Y_pred1
Y_pred1 = "The First Method: " +Emotions[int(Y_pred1[0])-1]
Y_pred2 = clf2.predict(TestX2)
Y_pred2 = "The Second Method: " +Emotions[int(Y_pred2[0])-1]
Y_pred3 = clf3.predict(TestX3)
Y_pred3 = "The Third Method: " +Emotions[int(Y_pred3[0])-1]

font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(I,Y_pred1,(10,20), font, 0.6,(255,0,0),2)
cv2.putText(I,Y_pred2,(10,50), font, 0.6,(0,255,0),2)
cv2.putText(I,Y_pred3,(10,80), font, 0.6,(0,0,255),2)

cv2.imwrite("Result.jpg",I)
cv2.waitKey(0)
cv2.destroyAllWindows()