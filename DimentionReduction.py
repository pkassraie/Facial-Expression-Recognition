import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from matplotlib import pyplot as plt

def DimentionReduction():
    X = np.genfromtxt('Features.csv', delimiter=',')[0:327,]
    y = np.genfromtxt('Labels.csv',delimiter=',')[0:327]
    target_names = ["Angry", "Disgust", "Fear", "Happy", "Sadness", "Surprised", "Contempt"]


    pca = PCA(n_components=10)
    X_pca = pca.fit(X).transform(X)
    np.savetxt("ReducedFeaturesPCA.csv",X_pca,delimiter=',')

    lda = LinearDiscriminantAnalysis(n_components=10)
    X_lda = lda.fit(X, y).transform(X)
    np.savetxt("ReducedFeaturesLDA.csv",X_lda,delimiter=',')

    # Percentage of variance explained for each components
    print('explained variance ratio: %s'
          % str(sum(pca.explained_variance_ratio_)))

    plt.figure()
    colors = ['navy', 'turquoise', 'darkorange', 'coral','darkseagreen','lightsage','lightskyblue']
    lw = 2

    for color, i, target_name in zip(colors, [0, 1, 2, 3, 4, 5, 6], target_names):
        plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], color=color, alpha=.8, lw=lw,
                    label=target_name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('PCA of Features')

    plt.figure()
    for color, i, target_name in zip(colors, [0, 1, 2, 3, 4, 5, 6], target_names):
        plt.scatter(X_lda[y == i, 0], X_lda[y == i, 1], alpha=.8, color=color,
                    label=target_name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('LDA of Features')

    plt.show()

DimentionReduction()