  # Facial-Expression-Recognition
Multi-class SVM classification

## Short Report on the Results
  A full report in persian has been uploaded (Images are informative though). The english version will be included.
## A Guide to the Files

### 0. Required Packages
  * Sklearn
  * Cv2
  * Numpy
  * Glob
  * Os
  * Scipy.io
  * Dlib
  
### 1. Modules for creating the Feature Vector
  * ExtractLandmarks: Extracts facial landmarks from an image according to the 1st proposed model
  * ExractNormalLandmarks: Extracts facial landmarks from an image according to the 2nd proposed model
  * ExtractLargeLandmarks: Extracts facial landmarks from an image according to the 3rd proposed model
  
### 2. Modules for creating the dataset
  * Method1:Creates a dataset for the feature vectors from the 1st model
  * Method2:Creates a dataset for the feature vectors from the 2nd model
  * Method3:Creates a dataset for the feature vectors from the 3rd model
  
### 3. Modules for dataset dimension reduction
  * DimentionReduction: Reduces the feature vector's dimension from 4556 to 10, using PCA & LDA
  
### 4. Modules for Testing the model
  * Testing: Tests a dataset with 6 types of classifiers, calculates the corresponding confusion matrix & presicion scores
  * TestOnImage: Takes an image or a video as in input, returns the Facial expression of the all the faces seen in the input
  
## Method Description
  Will soon be completed.
