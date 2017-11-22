from sklearn.externals import joblib
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.around(cm,decimals=2)
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')




Features = np.genfromtxt('Features.csv', delimiter=',')[0:327,]
Labels = np.genfromtxt('Labels.csv',delimiter=',')[0:327]

trainind = np.random.randint(0, high=Features.shape[0], size=int(Features.shape[0]*0.8))

TrainX = Features[trainind,:]
#TestX = np.delete(Features,trainind,0)
TestX = Features

TrainY = Labels[trainind]
#TestY = np.delete(Labels,trainind,0)
TestY = Labels

clf = LinearSVC()
#clf = AdaBoostClassifier()
#clf = SVC(kernel= 'poly',degree=5)
clf.fit(TrainX, TrainY)

joblib.dump(clf, 'Clf_LinSVC.pkl')

score = clf.score(TestX,TestY)
print "The Linear SVC Score is:", score


Y_pred = clf.predict(TestX)


cnf_matrix = confusion_matrix(TestY,Y_pred)
np.set_printoptions(precision=2)
class_names = ["Angry", "Disgust", "Fear", "Happy", "Sadness", "Surprised", "Contempt"]


plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,normalize=True,
                      title='Confusion matrix, with normalization - One vs. All Linear SVC')
plt.show()

clf2 = SVC(kernel='linear')
#clf2 = SVC() #RBF
#clf2 = SVC(kernel='poly',degree=7)
clf2.fit(TrainX, TrainY)

joblib.dump(clf2, 'Clf_SVC.pkl')

score2 = clf2.score(TestX,TestY)
print "The SVC Score is:", score2


Y_pred2 = clf2.predict(TestX)

cnf_matrix2 = confusion_matrix(TestY,Y_pred2)
np.set_printoptions(precision=2)



plt.figure()
plot_confusion_matrix(cnf_matrix2, classes=class_names,normalize=True,
                      title='Confusion matrix, with normalization - One vs. One Linear SVC')
plt.show()
