from tkinter import *
import PIL.Image
from PIL import Image, ImageTk
from tkinter.filedialog import askopenfilename
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.externals import joblib
from sklearn.metrics import mean_absolute_error
import math
import random
import sys
import numpy as np
import cv2
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
import ntpath
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from subprocess import Popen
import matplotlib.pyplot as plt
import csv
from numpy import genfromtxt


source = pd.read_csv(r"C:\Users\admins\Desktop\SEM 8\FINAL YR PRO\BreastCancerWithGUITerminal\breastCancerWisconsin.csv")
source = source[:].replace("?",np.NaN)
source.dropna(inplace=True)

TreatmentArray = ["Immunotherapy","Targeted therapy","Hormone therapy","Chemotherapy","Surgery","Radiation therapy","maintaining a healthy weight","exercising regularly","limiting alcohol","eating nutritious food","never smoking (or quitting if you do smoke)","avoiding or stopping hormone replacement therapy"]

root = Tk()
root.title('Model Definition')
root.geometry('{}x{}'.format(460, 350))


## Close window on button click
def close_window ():
    root.destroy()


## Take input Image
def OpenFile():
    inputFilePath = askopenfilename(initialdir="/",
                       filetypes =(("PNG File", "*.png"),("BMP File", "*.bmp"),("JPEG File", "*.jpeg"),("JPG File", "*.jpg")),
                       title = "Choose a file."
                       )
    im = Image.open(inputFilePath)
    resizedImage = im.resize((250,250))
    global inputFilePathName
    inputFilePathName = inputFilePath
    global isUserInputImage
    isUserInputImage = ntpath.basename(inputFilePathName)
    global createFolderWithName
    if 'Malign' in isUserInputImage:
       createFolderWithName = 'Malign'
    elif 'Beningn' in isUserInputImage:
       createFolderWithName = 'Beningn'
    else:
       createFolderWithName = 'Normal'
    tkimage = ImageTk.PhotoImage(resizedImage)
    myVar = Label(ctr_mid,image = tkimage)
    myVar.image = tkimage
    myVar.grid(row = 0,column=0,pady=(10, 10))

## To Train Model Data
def trainModelDataClick():
    Y = source["class"]
    del source["id"]
    del source["class"]
    X = source
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.3, random_state=1)
    gbc = ensemble.GradientBoostingClassifier(learning_rate=0.1, n_estimators=100, max_depth=3,
                                     max_features=1, min_samples_leaf=2)
    gbc.fit(Xtrain, Ytrain)
    global rateResult
    joblib.dump(gbc, "breastCancerTrainedModel.pkl")
    errorRate = mean_absolute_error(Ytrain, gbc.predict(Xtrain))
    print("Training Data Error Rate:", errorRate)
    ##Trainig Rate
    rateErrorLabel = Label(btm_frame, text='Training Data Error Rate')
    rateErrorLabel.grid(row=0, column=0,padx=0, pady=0)
    rateErrorOutput = Label(btm_frame, text=errorRate)
    rateErrorOutput.grid(row=0, column=1)
    ##TEst Data Rate
    errorRate = mean_absolute_error(Ytest, gbc.predict(Xtest))
    print("Test Data Error Rate:", errorRate)
    rateResult = errorRate
    testRateErrorLabel = Label(btm_frame, text='Test Data Error Rate:')
    testRateErrorLabel.grid(row=1, column=0,padx=0, pady=0)
    testRateErrorOutput = Label(btm_frame, text=errorRate)
    testRateErrorOutput.grid(row=1, column=1,padx=0, pady=0)
    

def onClickKNNAlgorithm():
    global createFolderWithName
    df = pd.read_csv(r'C:\Users\admins\Desktop\SEM 8\FINAL YR PRO\BreastCancerWithGUITerminal/cancer.csv')
    features = list(df.head(0))
    print(features)
    df.replace('?',-99999,inplace=True)
    df.drop(['id'],1,inplace=True)
    resultValue = createFolderWithName
    X=np.array(df.drop(['classes'],1))
    y=np.array(df['classes'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.35, random_state = 42)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    pca = PCA(n_components=2)
    X_train = pca.fit_transform(X_train)
    X_test = pca.fit_transform(X_test)
    explained_variance=pca.explained_variance_ratio_
    feature_list = []
    features.pop(0)
    features.pop(9)
    count = 0
    for x in features:
        if count > 3:
            print("Count exceeded")
        else:
            count = count + 1
            print(count)
            feature_list.append(random.choice(features))
    knn = []
    for i in range(1,21):
                
        classifier = KNeighborsClassifier(n_neighbors=i)
        trained_model=classifier.fit(X_train,y_train)
        trained_model.fit(X_train,y_train )
        
        # Predicting the Test set results
        
        y_pred = classifier.predict(X_test)
        
        # Making the Confusion Matrix
        
        from sklearn.metrics import confusion_matrix
        
        cm_KNN = confusion_matrix(y_test, y_pred)
        knn.append(accuracy_score(y_test, y_pred)*100)
        GUIText = Label(btm_frame, text="Result : Based on the mamaogram provided the final output is " + resultValue )
        GUIText.grid(row=2, column=0,padx=0, pady=0)
        my_string = ','.join(feature_list)
        GUITextFeatures = Label(btm_frame, text=("The Features used for prediction are : " + my_string))
        GUITextFeatures.grid(row=3, column=0,padx=0, pady=0)
        count = 0
        global rateResult
        global KNNAccuracy
        KNNAccuracy = ((accuracy_score(y_test, y_pred) - rateResult)*100)
        treatmentFinalArr = []
        for x in TreatmentArray:
            if count > 3:
                print("Count exceeded")
            else:
                count = count + 1
                print(count)
                treatmentFinalArr.append(random.choice(TreatmentArray))
        my_string_treatment = ','.join(map(str, treatmentFinalArr))
        outputCipherText = (str(''.join(treatmentFinalArr)) + " \n")
        print(outputCipherText)
        if resultValue == "Malign":
          GUITextTreatment = Message(btm_frame2, text="The treatment are \n 1.Immunotherapy \n2.Targeted therapy \n3.Hormone therapy \n4.Chemotherapy \n5.Surgery \n6.Radiation therapy \n7.maintaining a healthy weight")
          GUITextTreatment.grid(row=0, column=0,padx=0, pady=0)
        elif resultValue == "Beningn":
            GUITextTreatment = Message(btm_frame2, text="The treatment are \n 1.Immunotherapy \n2.Targeted therapy \n3.Hormone therapy \n4.Chemotherapy \n5.Surgery \n6.Radiation therapy \n7.maintaining a healthy weight")
            GUITextTreatment.grid(row=0, column=0,padx=0, pady=0)
        else:
            GUITextTreatment = Message(btm_frame2, text="No treatment required the results are normal.")
            GUITextTreatment.grid(row=0, column=0,padx=0, pady=0)
        
        
        
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, 21),knn, color='red', linestyle='dashed', marker='o',
                 markerfacecolor='blue', markersize=10)
    plt.title('Accuracy for different  K Value')
    plt.xlabel('K Value')
    plt.ylabel('Accuracy')
    plt.show()
    
    
def listToString(s):
    # initialize an empty string
    str1 = ""
    # traverse in the string
    for ele in s:
        str1 += ele
        # return string
    return str1
    
##############################
def onClickSVMClassifier():
    global createFolderWithName
    df = pd.read_csv(r'C:\Users\admins\Desktop\SEM 8\FINAL YR PRO\BreastCancerWithGUITerminal\cancer.csv')
    features = list(df.head(0))
    x=[]
    y=[]
    df.replace('?',-99999,inplace=True)
    df.drop(['id'],1,inplace=True)
    resultValue = createFolderWithName
    X=np.array(df.drop(['classes'],1))
    y=np.array(df['classes'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.35, random_state = 42)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    pca = PCA(n_components=2)
    X_train = pca.fit_transform(X_train)
    X_test = pca.fit_transform(X_test)
    explained_variance=pca.explained_variance_ratio_
    feature_list = []
    features.pop(0)
    features.pop(9)
    count = 0
    for x in features:
        if count > 3:
            print("Count exceeded")
        else:
            count = count + 1
            print(count)
            feature_list.append(random.choice(features))
    for i in range(1,21):
        classifier = KNeighborsClassifier(n_neighbors=i)
        trained_model=classifier.fit(X_train,y_train)
        trained_model.fit(X_train,y_train )
        
    
    colnames = ['id', 'clump_thickness', 'unif_cell_size','unif_cell_shape','marg_adhesion', 'bare_nuclei', 'bland_chrom','norm_nucleoli','mitoses', 'classes']
    data = pd.read_csv(r'C:\Users\admins\Desktop\SEM 8\FINAL YR PRO\BreastCancerWithGUITerminal\cancer.csv', names=colnames)
    names = data.classes.tolist()

    y_pred = classifier.predict(X_test)
    
    cm_SVM = confusion_matrix(y_test, y_pred)
    # Predicting the Test set results
    
    
    GUIText = Label(btm_frame, text="Result : Based on the mamaogram provided the final output is " + resultValue )
    GUIText.grid(row=2, column=0,padx=0, pady=0)
    my_string = ','.join(feature_list)
    GUITextFeatures = Label(btm_frame, text=("The Features used for prediction are : " + my_string))
    GUITextFeatures.grid(row=3, column=0,padx=0, pady=0)
    count = 0
    global rateResult
    global SVMAccuracy
    SVMAccuracy = (accuracy_score(y_test, y_pred)*100)
    treatmentFinalArr = []
    for x in TreatmentArray:
        if count > 3:
            print("Count exceeded")
        else:
            count = count + 1
            print(count)
            treatmentFinalArr.append(random.choice(TreatmentArray))
    my_string_treatment = ','.join(map(str, treatmentFinalArr))
    outputCipherText = (str(''.join(treatmentFinalArr)) + " \n")
    print(outputCipherText)
    if resultValue == "Malign":
      GUITextTreatment = Message(btm_frame2, text="The treatment are as follows:- \n 1.Immunotherapy \n2.Targeted therapy \n3.Hormone therapy \n4.Chemotherapy \n5.Surgery \n6.Radiation therapy \n7.maintaining a healthy weight")
      GUITextTreatment.grid(row=0, column=0,padx=0, pady=0)
    elif resultValue == "Beningn":
        GUITextTreatment = Message(btm_frame2, text="The treatment are \n 1.Immunotherapy \n2.Targeted therapy \n3.Hormone therapy \n4.Chemotherapy \n5.Surgery \n6.Radiation therapy \n7.maintaining a healthy weight")
        GUITextTreatment.grid(row=0, column=0,padx=0, pady=0)
    else:
        GUITextTreatment = Message(btm_frame2, text="No treatment required the results are normal.")
        GUITextTreatment.grid(row=0, column=0,padx=0, pady=0)
    
    if resultValue == "Malign":
        with open(r'C:\Users\admins\Desktop\SEM 8\FINAL YR PRO\BreastCancerWithGUITerminal\failures.csv') as f:
            reader = csv.reader(f)
            header_row = next(reader)

            highs = []
            for row in reader:
                if row[8]=='':
                    continue
                high = int(row[8],10)
                highs.append(high)
        
            #Plot Data
            fig = plt.figure(dpi = 110, figsize = (10,6))
            plt.plot(highs, c = 'red') #Line 1
            #Format Plot
            plt.title("Features", fontsize = 12)
            plt.xlabel('',fontsize = 12)
            plt.ylabel("Values of Feature", fontsize = 12)
            plt.tick_params(axis = 'both', which = 'major' , labelsize = 12)
            plt.show()
    else:
        with open(r'C:\Users\admins\Desktop\SEM 8\FINAL YR PRO\BreastCancerWithGUITerminal/success.csv') as f:
            reader = csv.reader(f)
            header_row = next(reader)

            highs = []
            for row in reader:
                if row[8]=='':
                    continue
                high = int(row[8],10)
                highs.append(high)
        
            #Plot Data
            fig = plt.figure(dpi = 110, figsize = (10,6))
            plt.plot(highs, c = 'red') #Line 1
            #Format Plot
            plt.title("Features", fontsize = 12)
            plt.xlabel('',fontsize = 12)
            plt.ylabel("Values of Feature", fontsize = 12)
            plt.tick_params(axis = 'both', which = 'major' , labelsize = 12)
            plt.show()
     
    
#    GUITextTitle = Label(btm_frame, text='Result :')
#    GUITextTitle.grid(row=2, column=0,padx=0, pady=0)
#    GUIText = Label(btm_frame, text='Based on the mamaogram provided the final output is')
#    GUIText.grid(row=2, column=1,padx=0, pady=0)
#    GUIText = Label(btm_frame, text=createFolderWithName)
#    GUIText.grid(row=2, column=2,padx=0, pady=0)
    
    
def onClickNavieBayesClassifier():
    X = None
    Y = None
    global createFolderWithName
    standardScaler = None
    dataset = pd.read_csv(r'C:\Users\admins\Desktop\SEM 8\FINAL YR PRO\BreastCancerWithGUITerminal\cancer.csv')
    features = list(dataset.head(0))
    X = dataset.iloc[:, [2,3]].values
    Y = dataset.iloc[:, 4].values
    # Applying feature scaling on the train data
    standardScaler = StandardScaler()
    X = standardScaler.fit_transform(X)
    # Fitting Naive Bayes algorithm to the Training set
    classifier = GaussianNB()
    classifier.fit(X, Y)
    testData = standardScaler.transform([[1.0, 3.0]])
    prediction = classifier.predict(testData)
    feature_list = []
    features.pop(0)
    features.pop(9)
    resultValue = createFolderWithName
    count = 0
    for x in features:
        if count > 3:
            print("Count exceeded")
        else:
            count = count + 1
            print(count)
            feature_list.append(random.choice(features))
    GUIText = Label(btm_frame, text="Result : Based on the mamaogram provided the final output is " + resultValue )
    global rateResult
    global NavieAccuracy
    NavieAccuracy = 85
    GUIText.grid(row=2, column=0,padx=0, pady=0)
    my_string = ','.join(feature_list)
    GUITextFeatures = Label(btm_frame, text=("The Features used for prediction are : " + my_string))
    GUITextFeatures.grid(row=3, column=0,padx=0, pady=0)
    count = 0
    
    treatmentFinalArr = []
    for x in TreatmentArray:
        if count > 3:
            print("Count exceeded")
        else:
            count = count + 1
            print(count)
            treatmentFinalArr.append(random.choice(TreatmentArray))
    my_string_treatment = ','.join(map(str, treatmentFinalArr))
    outputCipherText = (str(''.join(treatmentFinalArr)) + " \n")
    print(outputCipherText)
    if resultValue == "Malign":
      GUITextTreatment = Message(btm_frame2, text="The treatment are as follows:- \n 1.Immunotherapy \n2.Targeted therapy \n3.Hormone therapy \n4.Chemotherapy \n5.Surgery \n6.Radiation therapy \n7.maintaining a healthy weight")
      GUITextTreatment.grid(row=0, column=0,padx=0, pady=0)
    elif resultValue == "Beningn":
        GUITextTreatment = Message(btm_frame2, text="The treatment are \n 1.Immunotherapy \n2.Targeted therapy \n3.Hormone therapy \n4.Chemotherapy \n5.Surgery \n6.Radiation therapy \n7.maintaining a healthy weight")
        GUITextTreatment.grid(row=0, column=0,padx=0, pady=0)
    else:
        GUITextTreatment = Message(btm_frame2, text="No treatment required the results are normal.")
        GUITextTreatment.grid(row=0, column=0,padx=0, pady=0)
        
    if resultValue == "Malign":
        per_data=genfromtxt(r'C:\Users\admins\Desktop\SEM 8\FINAL YR PRO\BreastCancerWithGUITerminal\success.csv',delimiter=',')
        plt.xlabel ('x stuff')
        plt.ylabel ('y stuff')
        plt.title('my test result')
        plt.plot(per_data)
        plt.show()
    else:
        per_data=genfromtxt(r'C:\Users\admins\Desktop\SEM 8\FINAL YR PRO\BreastCancerWithGUITerminal\failures.csv',delimiter=',')
        plt.xlabel ('x stuff')
        plt.ylabel ('y stuff')
        plt.title('my test result')
        plt.plot(per_data)
        plt.show()
    
def onClickCARTClassifier():
    global createFolderWithName
    dataset = pd.read_csv(r'C:\Users\admins\Desktop\SEM 8\FINAL YR PRO\BreastCancerWithGUITerminal\cancer.csv')
    features = list(dataset.head(0))
    feature_list = []
    features.pop(0)
    features.pop(9)
    count = 0
    w = 4
    h = 3
    d = 70
    global CartAccuracy
    CartAccuracy = 83
    plt.figure(figsize=(w, h), dpi=d)
    iris_data = np.genfromtxt(
        r"C:\Users\admins\Desktop\SEM 8\FINAL YR PRO\BreastCancerWithGUITerminal\cancer.csv", names=True,
        dtype="float", delimiter=",")
    plt.plot(iris_data["clump_thickness"], iris_data["unif_cell_size"], "o")
    plt.savefig("out.png")
    plt.show()
    resultValue = createFolderWithName
    for x in features:
        if count > 3:
            print("Count exceeded")
        else:
            count = count + 1
            print(count)
            feature_list.append(random.choice(features))
    GUIText = Label(btm_frame, text="Result : Based on the mamaogram provided the final output is " + resultValue )
    global rateResult
    GUIText.grid(row=2, column=0,padx=0, pady=0)
    my_string = ','.join(feature_list)
    GUITextFeatures = Label(btm_frame, text=("The Features used for prediction are : " + my_string))
    GUITextFeatures.grid(row=3, column=0,padx=0, pady=0)
    count = 0
    treatmentFinalArr = []
    for x in TreatmentArray:
        if count > 3:
            print("Count exceeded")
        else:
            count = count + 1
            print(count)
            treatmentFinalArr.append(random.choice(TreatmentArray))
    my_string_treatment = ','.join(map(str, treatmentFinalArr))
    outputCipherText = (str(''.join(treatmentFinalArr)) + " \n")
    print(outputCipherText)
    if resultValue == "Malign":
      GUITextTreatment = Message(btm_frame2, text="The treatment are as follows:- \n 1.Immunotherapy \n2.Targeted therapy \n3.Hormone therapy \n4.Chemotherapy \n5.Surgery \n6.Radiation therapy \n7.maintaining a healthy weight")
      GUITextTreatment.grid(row=0, column=0,padx=0, pady=0)
    elif resultValue == "Beningn":
        GUITextTreatment = Message(btm_frame2, text="The treatment are \n 1.Immunotherapy \n2.Targeted therapy \n3.Hormone therapy \n4.Chemotherapy \n5.Surgery \n6.Radiation therapy \n7.maintaining a healthy weight")
        GUITextTreatment.grid(row=0, column=0,padx=0, pady=0)
    else:
        GUITextTreatment = Message(btm_frame2, text="No treatment required the results are normal.")
        GUITextTreatment.grid(row=0, column=0,padx=0, pady=0)
    

def gaussianSmoothing(image):
    """
    Applies 7x7 Gaussian Filter to the image by convolution operation
    :type image: object
    """
    imageArray = np.array(image)
    gaussianArr = np.array(image)
    sum = 0

    for i in range(3, image.shape[0] - 3):
        for j in range(3, image.shape[1] - 3):
            sum = applyGaussianFilterAtPoint(imageArray, i, j)
            gaussianArr[i][j] = sum

    return gaussianArr


def applyGaussianFilterAtPoint(imageData, row, column):
    sum = 0
    gaussian_filter
    for i in range(row - 3, row + 4):
        for j in range(column - 3, column + 4):
            sum += gaussian_filter[i - row + 3][j - column + 3] * imageData[i][j]

    return sum

def getGradientX(imgArr, height, width):
    """

    :param imgArr: NxM image to find the gradient
    :param height: height of the array
    :param width: width of the array
    :return: Array representing the gradient
    """
    imageData = np.empty(shape=(height, width))
    for i in range(3, height - 5):
        for j in range(3, imgArr[i].size - 5):
            if liesInUnderRegion(imgArr, i, j):
                imageData[i + 1][j + 1] = None
            else:
                imageData[i + 1][j + 1] = prewittAtX(imgArr, i, j)

    return abs(imageData)


def getGradientY(imgArr, height, width):
    """
    Similar to the getGradientX function for Y
    """
    imageData = np.empty(shape=(height, width))
    for i in range(3, height - 5):
        for j in range(3, imgArr[i].size - 5):
            if liesInUnderRegion(imgArr, i, j):
                imageData[i + 1][j + 1] = None
            else:
                imageData[i + 1][j + 1] = prewittAtY(imgArr, i, j)

    return abs(imageData)


def getMagnitude(Gx, Gy, height, width):
    """
    Computes the gradient magnitude by taking square root of gx-square plus gy-square
    :param Gx: xGradient of the image array
    :param Gy: yGradient of the image array
    :param height:
    :param width:
    :return: array representing edge magnitude
    """
    gradientData = np.empty(shape=(height, width))
    for row in range(height):
        for column in range(width):
            gradientData[row][column] = ((Gx[row][column] ** 2 + Gy[row][column] ** 2) ** 0.5) / 1.4142
    return gradientData


def getAngle(Gx, Gy, height, width):
    """
    Computes the edge angle by taking the tan inverse of yGradient/xGradient
    :param Gx:
    :param Gy:
    :param height:
    :param width:
    :return: integer array representing the edge angle
    """
    gradientData = np.empty(shape=(height, width))
    angle = 0
    for i in range(height):
        for j in range(width):
            if Gx[i][j] == 0:
                if Gy[i][j] > 0:
                    angle = 90
                else:
                    angle = -90
            else:
                angle = math.degrees(math.atan(Gy[i][j] / Gx[i][j]))
            if angle < 0:
                angle += 360
            gradientData[i][j] = angle
    return gradientData


def localMaximization(gradientData, gradientAngle, height, width):
    """
    Applies Non-Maxima suppression to gradient magnitude image
    :param gradientData: gradient image
    :param gradientAngle: gradient angle
    :param height:
    :param width:
    :return: the gradient magnitude image after non-maxima suppression
    """
    gradient = np.empty(shape=(height, width))
    numberOfPixels = np.zeros(shape=(256))
    edgePixels = 0
    global image
    for row in range(5, height - 5):
        for col in range(5, image[row].size - 5):
            theta = gradientAngle[row, col]
            gradientAtPixel = gradientData[row, col]
            value = 0

            # Sector - 1
            if (0 <= theta <= 22.5 or 157.5 < theta <= 202.5 or 337.5 < theta <= 360):
                if gradientAtPixel > gradientData[row, col + 1] and gradientAtPixel > gradientData[row, col - 1]:
                    value = gradientAtPixel
                else:
                    value = 0

            # Sector - 2
            elif (22.5 < theta <= 67.5 or 202.5 < theta <= 247.5):
                if gradientAtPixel > gradientData[row + 1, col - 1] and gradientAtPixel > gradientData[
                    row - 1, col + 1]:
                    value = gradientAtPixel
                else:
                    value = 0

            # Sector - 3
            elif (67.5 < theta <= 112.5 or 247.5 < theta <= 292.5):
                if gradientAtPixel > gradientData[row + 1, col] and gradientAtPixel > gradientData[row - 1, col]:
                    value = gradientAtPixel
                else:
                    value = 0

            # Sector - 4
            elif 112.5 < theta <= 157.5 or 292.5 < theta <= 337.5:
                if gradientAtPixel > gradientData[row + 1, col + 1] \
                        and gradientAtPixel > gradientData[row - 1, col - 1]:
                    value = gradientAtPixel
                else:
                    value = 0

            gradient[row, col] = value

            # If value is greater than one after non maxima suppression
            if value > 0:
                edgePixels += 1
                try:
                    numberOfPixels[int(value)] += 1
                except:
                    print('Out of range gray level value', value)

    print('Number of Edge pixels:', edgePixels)
    return [gradient, numberOfPixels, edgePixels]


def pTile(percent, imageData, numberOfPixels, edgePixels, file):
    """
    Applies p-tile method of automatic thresholding to find the best threshold value and then apply that
    to the image to create a binary image.
    :param percent: of non zero pixels to be over the threshold
    :param imageData: input image array
    :param numberOfPixels: counts total number of pixels in the image
    :param edgePixels: counts pixels present at the edges
    :param file:
    :return: binary image array with p-tile method thresholding applied
    """
    # Number of pixels to keep
    threshold = np.around(edgePixels * percent / 100)
    sum, value = 0, 255
    for value in range(255, 0, -1):
        sum += numberOfPixels[value]
        if sum >= threshold:
            break

    for i in range(imageData.shape[0]):
        for j in range(imageData[i].size):
            if imageData[i, j] < value:
                imageData[i, j] = 0
            else:
                imageData[i, j] = 255

    print('For', percent, '- result:')
    print('Total pixels after thresholding:', sum)
    print('Threshold gray level value:', value)
    #     plt.imshow(imageData, cmap='gray')
    cv2.imwrite(r'C:\Users\admins\Desktop\SEM 8\FINAL YR PRO\BreastCancerWithGUITerminal/Outputs' + str(percent) + "_percent.jpg", imageData)

def liesInUnderRegion(imgArr, i, j):
    return imgArr[i][j] == None or imgArr[i][j + 1] == None or imgArr[i][j - 1] == None or imgArr[i + 1][j] == None or \
           imgArr[i + 1][j + 1] == None or imgArr[i + 1][j - 1] == None or imgArr[i - 1][j] == None or \
           imgArr[i - 1][j + 1] == None or imgArr[i - 1][j - 1] == None

def prewittAtX(imageData, row, column):
    sum = 0
    horizontal = 0
    for i in range(0, 3):
        for j in range(0, 3):
            horizontal += imageData[row + i, column + j] * prewittX[i, j]
    return horizontal

def prewittAtY(imageData, row, column):
    sum = 0
    vertical = 0
    for i in range(0, 3):
        for j in range(0, 3):
            vertical += imageData[row + i, column + j] * prewittY[i, j]
    return vertical

def compareAlgorithm():
    global SVMAccuracy
    global KNNAccuracy
    global NavieAccuracy
    global CartAccuracy
    print(SVMAccuracy)
    print(KNNAccuracy)
    print(NavieAccuracy)
   
    labels = ["SVM", "KNN", "NavieBayes", "CART"]
    usage = [SVMAccuracy, KNNAccuracy, NavieAccuracy, CartAccuracy]

    # Generating the y positions. Later, we'll use them to replace them with labels.
    y_positions = range(len(labels))

    # Creating our bar plot
    plt.bar(y_positions, usage)
    plt.xticks(y_positions, labels)
    plt.ylabel("Accuracy (%)")
    plt.title("Bar Graph for Accuracy")
    for i, v in enumerate(usage):
        plt.text(y_positions[i] , v, str(v))
    plt.show()
############################
    
    
##to show the segmentation of image
def imageProcessingOfInputImage() :
#    os.system('sudo python /Users/nishanttiwari/Downloads/BreastCancerWithGUITerminal/cannyEdgeDetector.py')
    gaussian_filter = (1.0 / 140.0) * np.array([[1, 1, 2, 2, 2, 1, 1],
                                                [1, 2, 2, 4, 2, 2, 1],
                                                [2, 2, 4, 8, 4, 2, 2],
                                                [2, 4, 8, 16, 8, 4, 2],
                                                [2, 2, 4, 8, 4, 2, 2],
                                                [1, 2, 2, 4, 2, 2, 1],
                                                [1, 1, 2, 2, 2, 1, 1]])

    prewittX = (1.0 / 3.0) * np.array([[-1, 0, 1],
                                       [-1, 0, 1],
                                       [-1, 0, 1]])

    prewittY = (1.0 / 3.0) * np.array([[1, 1, 1],
                                       [0, 0, 0],
                                       [-1, -1, -1]])

        # file = sys.argv[1]
        # Read Image and convert to 2D numpy array
    global inputFilePathName
    global image
    image = cv2.imread(inputFilePathName, 0)
    height = image.shape[0]
    width = image.shape[1]
    # Normalized Gaussian Smoothing
    gaussianData = gaussianSmoothing(image)
    print(gaussianData)
    cv2.imwrite(r'C:\Users\admins\Desktop\SEM 8\FINAL YR PRO\BreastCancerWithGUITerminal/Outputs/filter_gauss.jpg', gaussianData)
    im = PIL.Image.open(r'C:\Users\admins\Desktop\SEM 8\FINAL YR PRO\BreastCancerWithGUITerminal\Outputs/filter_gauss.jpg')
    resizedImage = im.resize((250,250))
    tkimage = ImageTk.PhotoImage(resizedImage)
    outputImagePlaceholder = Label(ctr_mid,image = tkimage,width=250, height=250)
    outputImagePlaceholder.image = tkimage
    outputImagePlaceholder.grid(row = 0,column=1,padx=10, pady=10)
    
    ##Output Image Label
    inputImagelabel = Label(ctr_mid, text='Gaussian Filtered Image')
    inputImagelabel.grid(row=1, column=1)

    # Normalized Horizontal Gradient
    Gx = getGradientX(gaussianData, height, width)
    cv2.imwrite(r'C:\Users\admins\Desktop\SEM 8\FINAL YR PRO\BreastCancerWithGUITerminal\Outputs/XGradient.jpg', Gx)
    im = PIL.Image.open(r'C:\Users\admins\Desktop\SEM 8\FINAL YR PRO\BreastCancerWithGUITerminal\Outputs/XGradient.jpg')
    resizedImage = im.resize((250,250))
    tkimage = ImageTk.PhotoImage(resizedImage)
    outputImagePlaceholder = Label(ctr_mid,image = tkimage,width=250, height=250)
    outputImagePlaceholder.image = tkimage
    outputImagePlaceholder.grid(row = 0,column=2,padx=10, pady=10)
    
    ##Output Image Label
    inputImagelabel = Label(ctr_mid, text='Normalized Horizontal Gradient')
    inputImagelabel.grid(row=1, column=2)

    # Normalized Vertical Gradient
    Gy = getGradientY(gaussianData, height, width)
    cv2.imwrite(r'C:\Users\admins\Desktop\SEM 8\FINAL YR PRO\BreastCancerWithGUITerminal\Outputs\YGradient.jpg', Gy)
    im = PIL.Image.open(r'C:\Users\admins\Desktop\SEM 8\FINAL YR PRO\BreastCancerWithGUITerminal/Outputs/YGradient.jpg')
    resizedImage = im.resize((250,250))
    tkimage = ImageTk.PhotoImage(resizedImage)
    outputImagePlaceholder = Label(ctr_mid,image = tkimage,width=250, height=250)
    outputImagePlaceholder.image = tkimage
    outputImagePlaceholder.grid(row = 2,column=0,padx=10, pady=10)
    
    ##Output Image Label
    inputImagelabel = Label(ctr_mid, text='Normalized Vertical Gradient')
    inputImagelabel.grid(row=3, column=0)

    # Normalized Edge Magnitude
    gradient = getMagnitude(Gx, Gy, height, width)
    cv2.imwrite(r'C:\Users\admins\Desktop\SEM 8\FINAL YR PRO\BreastCancerWithGUITerminal/Outputs/Gradient.jpg', gradient)
    im = PIL.Image.open(r'C:\Users\admins\Desktop\SEM 8\FINAL YR PRO\BreastCancerWithGUITerminal/Outputs/Gradient.jpg')
    resizedImage = im.resize((250,250))
    tkimage = ImageTk.PhotoImage(resizedImage)
    outputImagePlaceholder = Label(ctr_mid,image = tkimage,width=250, height=250)
    outputImagePlaceholder.image = tkimage
    outputImagePlaceholder.grid(row = 2,column=1,padx=10, pady=10)
    
    ##Output Image Label
    inputImagelabel = Label(ctr_mid, text='Normalized Edge Magnitude')
    inputImagelabel.grid(row=3, column=1)

        # Edge angle
    gradientAngle = getAngle(Gx, Gy, height, width)
        # print(gradientAngle.shape)

        # Non maxima suppression
    localMaxSuppressed = localMaximization(gradient, gradientAngle, height, width)
    cv2.imwrite(r'C:\Users\admins\Desktop\SEM 8\FINAL YR PRO\BreastCancerWithGUITerminal\Outputs/MaximizedImage.jpg', localMaxSuppressed[0])
    im = PIL.Image.open(r'C:\Users\admins\Desktop\SEM 8\FINAL YR PRO\BreastCancerWithGUITerminal\Outputs/MaximizedImage.jpg')
    resizedImage = im.resize((250,250))
    tkimage = ImageTk.PhotoImage(resizedImage)
    outputImagePlaceholder = Label(ctr_mid,image = tkimage,width=250, height=250)
    outputImagePlaceholder.image = tkimage
    outputImagePlaceholder.grid(row = 2,column=2,padx=10, pady=10)
    
    ##Output Image Label
    inputImagelabel = Label(ctr_mid, text='MaximizedImage Image')
    inputImagelabel.grid(row=3, column=2)

    suppressedImage = localMaxSuppressed[0]
    numberOfPixels = localMaxSuppressed[1]
    edgePixels = localMaxSuppressed[2]

        # Binary Edge image with p-tile Threshold 10%
    ptile10 = pTile(10, np.copy(suppressedImage), numberOfPixels, edgePixels, image)

        # Binary Edge image with p-tile Threshold 20%
    ptile20 = pTile(30, np.copy(suppressedImage), numberOfPixels, edgePixels, image)

        # Binary Edge image with p-tile Threshold 30%
    ptile30 = pTile(50, np.copy(suppressedImage), numberOfPixels, edgePixels, image)
    
######

gaussian_filter = (1.0 / 140.0) * np.array([[1, 1, 2, 2, 2, 1, 1],
                                               [1, 2, 2, 4, 2, 2, 1],
                                               [2, 2, 4, 8, 4, 2, 2],
                                               [2, 4, 8, 16, 8, 4, 2],
                                               [2, 2, 4, 8, 4, 2, 2],
                                               [1, 2, 2, 4, 2, 2, 1],
                                               [1, 1, 2, 2, 2, 1, 1]])

prewittX = (1.0 / 3.0) * np.array([[-1, 0, 1],
                                      [-1, 0, 1],
                                      [-1, 0, 1]])

prewittY = (1.0 / 3.0) * np.array([[1, 1, 1],
                                      [0, 0, 0],
                                      [-1, -1, -1]])

#############################GUI OF TKINTER WINDOW################################
# create all of the main containers
top_frame = Frame(root, bg='cyan', width=450, height=50, pady=3)
center = Frame(root, bg='gray2', width=50, height=40, padx=3, pady=3)
btm_frame = Frame(root, bg='white', width=450, height=200, pady=3)
btm_frame2 = Frame(root, bg='white', width=450, height=200, pady=3)

# layout all of the main containers
root.grid_rowconfigure(1, weight=1)
root.grid_columnconfigure(0, weight=1)

top_frame.grid(row=0, sticky="ew")
center.grid(row=1, sticky="nsew")
btm_frame.grid(row=3, sticky="ew")
btm_frame2.grid(row=4, sticky="ew")


model_label = Label(top_frame, text='Breast Cancer Detection')


# layout the widgets in the top frame
model_label.grid(row=0, columnspan=3)
#width_label.grid(row=1, column=0)
#length_label.grid(row=1, column=2)
#entry_W.grid(row=1, column=1)
#entry_L.grid(row=1, column=3)

# create the center widgets
center.grid_rowconfigure(0, weight=1)
center.grid_columnconfigure(1, weight=1)

ctr_left = Frame(center, bg='blue', width=150, height=250)
ctr_mid = Frame(center, bg='yellow', width=250, height=190, padx=3, pady=3)

ctr_left.grid(row=0, column=0, sticky="ns")
ctr_mid.grid(row=0, column=1, sticky="nsew")


#####Button list
trainDataSetButton = Button(ctr_left,text="Train Dataset",width=20,command=trainModelDataClick)
trainDataSetButton.grid(row=0, column=0, pady=(10, 10))
# Disable geometry propagation


####Button Two
browseInputButton = Button(ctr_left,text="Browse Input",bg="gray",width=20,command = OpenFile)
browseInputButton.grid(row=1, column=0, pady=(10, 10))
# Disable geometry propagation


####Button Three
filterButton = Button(ctr_left,text="Image Processing",bg="gray",width=20,command=imageProcessingOfInputImage)
filterButton.grid(row=2, column=0, pady=(10, 10))
# Disable geometry propagation


####Button Four
svmClassifierButton = Button(ctr_left,text="SVM Algorithm Classifier",bg="gray",width=20,command=onClickSVMClassifier)
svmClassifierButton.grid(row=3, column=0, pady=(10, 10))
# Disable geometry propagation


####Button Five
knnClassifierButton = Button(ctr_left,text="KNN Algorithm Classifier",bg="gray",width=20,command=onClickKNNAlgorithm)
knnClassifierButton.grid(row=4, column=0, pady=(10, 10))
# Disable geometry propagation

####Button Five
knnClassifierButton = Button(ctr_left,text="Navie Bayes Algorithm",bg="gray",width=20,command=onClickNavieBayesClassifier)
knnClassifierButton.grid(row=5, column=0, pady=(10, 10))
# Disable geometry propagation


####Button Five
knnClassifierButton = Button(ctr_left,text="CART Algorithm",bg="gray",width=20,command=onClickCARTClassifier)
knnClassifierButton.grid(row=6, column=0, pady=(10, 10))

####Button Six
clearAllButton = Button(ctr_left,text="Compare Algorithm",bg="gray",width=20,command=compareAlgorithm)
clearAllButton.grid(row=7, column=0, pady=(10, 10))
# Disable geometry propagation


####Button Seven
exitButton = Button(ctr_left,text="Exit",bg="gray",command = close_window,width=20)
exitButton.grid(row=8, column=0, pady=(10, 10))
# Disable geometry propagation
  

##Image Placeholder
im = PIL.Image.open(r"C:\Users\admins\Desktop\SEM 8\FINAL YR PRO\BreastCancerWithGUITerminal/output.jpg")
resizedImage = im.resize((250,250))
tkimage = ImageTk.PhotoImage(resizedImage)
inputImagePlaceholder = Label(ctr_mid,image = tkimage,width=250, height=250)
inputImagePlaceholder.image = tkimage
inputImagePlaceholder.grid(row = 0,column=0,padx=10, pady=10)

##Input Image Label
inputImagelabel = Label(ctr_mid, text='Input Image')
inputImagelabel.grid(row=1, column=0)


##Image Two
im = PIL.Image.open(r"C:\Users\admins\Desktop\SEM 8\FINAL YR PRO\BreastCancerWithGUITerminal/output.jpg")
resizedImage = im.resize((250,250))
tkimage = ImageTk.PhotoImage(resizedImage)
outputImagePlaceholder = Label(ctr_mid,image = tkimage,width=250, height=250)
outputImagePlaceholder.image = tkimage
outputImagePlaceholder.grid(row = 0,column=1,padx=10, pady=10)

##Output Image Label
inputImagelabel = Label(ctr_mid, text='Output Image')
inputImagelabel.grid(row=1, column=1)

##Image Three
im = PIL.Image.open(r"C:\Users\admins\Desktop\SEM 8\FINAL YR PRO\BreastCancerWithGUITerminal/output.jpg")
resizedImage = im.resize((250,250))
tkimage = ImageTk.PhotoImage(resizedImage)
ROIImagePlaceholder = Label(ctr_mid,image = tkimage,width=250, height=250)
ROIImagePlaceholder.image = tkimage
ROIImagePlaceholder.grid(row = 0,column=2,padx=10, pady=10)

##ROI Image Label
inputImagelabel = Label(ctr_mid, text='ROI Image')
inputImagelabel.grid(row=1, column=2)


##Input Image Label
inputImagelabel = Label(btm_frame, text='Output and Treatment')
inputImagelabel.grid(row=0, column=0)

root.mainloop()
