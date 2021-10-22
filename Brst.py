import pandas as pd
from PIL import Image
from math import pi
import matplotlib.pyplot as plt
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder #to transform data into numerical
from sklearn.preprocessing import StandardScaler #for normalizing data
from sklearn.metrics import confusion_matrix #Confusion matrix for prediction accuracy
from sklearn.metrics import accuracy_score #accuracy score
import numpy as np
import streamlit as st
import streamlit_theme as stt
st.set_option('deprecation.showPyplotGlobalUse', False) #remove warning!

result = "No results yet."

stt.set_theme({'primary': '#0f87ff'})

st.sidebar.image("breast.png", width=400)
st.sidebar.subheader("Software is based on machine learning algorithms.")
st.sidebar.text(" ")

#Dataset is Wisconsin Breast cancer dataset
b_data = pd.read_csv("data.csv")


X = b_data.iloc[:, 2:32] #use iloc!
Y = b_data.iloc[:, 1]
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y) #M and B to 1 and 0

#Check if data missing or null

#dataset.isnull().sum()
#dataset.isna().sum()

#Split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

#Normalize data
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

######################################MODELS!!!!#############################

#Using Logistic Regression Algorithm to the Training Set
from sklearn.linear_model import LogisticRegression
classifierLR = LogisticRegression(random_state = 0)
classifierLR.fit(X_train, Y_train)

#Using KNeighborsClassifier Method of neighbors class to use Nearest Neighbor algorithm
from sklearn.neighbors import KNeighborsClassifier
classifierKN = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifierKN.fit(X_train, Y_train)

#Using SVC method of svm class to use Support Vector Machine Algorithm
from sklearn.svm import SVC
classifierSVM = SVC(kernel = 'linear', random_state = 0)
classifierSVM.fit(X_train, Y_train)

#Using SVC method of svm class to use Kernel SVM Algorithm
from sklearn.svm import SVC
classifierSVC = SVC(kernel = 'rbf', random_state = 0)
classifierSVC.fit(X_train, Y_train)

#Using GaussianNB method of naïve_bayes class to use Naïve Bayes Algorithm
from sklearn.naive_bayes import GaussianNB
classifierGNB = GaussianNB()
classifierGNB.fit(X_train, Y_train)

#Using DecisionTreeClassifier of tree class to use Decision Tree Algorithm
from sklearn.tree import DecisionTreeClassifier
classifierDT = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifierDT.fit(X_train, Y_train)

#Using RandomForestClassifier method of ensemble class to use Random Forest Classification algorithm
from sklearn.ensemble import RandomForestClassifier
classifierRF = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifierRF.fit(X_train, Y_train)

##########################################PREDICTIONS###################################
Y_predLR = classifierLR.predict(X_test)
Y_predKN = classifierKN.predict(X_test)
Y_predSVM = classifierSVM.predict(X_test)
Y_predSVC = classifierSVC.predict(X_test)
Y_predGNB = classifierGNB.predict(X_test)
Y_predDT = classifierDT.predict(X_test)
Y_predRF = classifierRF.predict(X_test)

acLR = round(accuracy_score(Y_test, Y_predLR)*100, 2)
acKN = round(accuracy_score(Y_test, Y_predKN)*100, 2)
acSVM = round(accuracy_score(Y_test, Y_predSVM)*100, 2)
acSVC = round(accuracy_score(Y_test, Y_predSVC)*100, 2)
acGNB = round(accuracy_score(Y_test, Y_predGNB)*100, 2)
acDT = round(accuracy_score(Y_test, Y_predDT)*100, 2)
acRF = round(accuracy_score(Y_test, Y_predRF)*100, 2)

#########################################################################################

st.sidebar.subheader("Based on the input data, algorithms can classify whether the data suggests that the tumor is malignant or benign.")
st.sidebar.write("Dataset consists of more than 500 medical records.")
st.sidebar.subheader("Different algorithms have different accuracies:")
st.sidebar.write("")
st.sidebar.write("Logistic regression: ", acLR, " %")
st.sidebar.write("Nearest neighbor: ", acKN, " %")
st.sidebar.write("SVM: ", acSVM, " %")
st.sidebar.write("Kernel SVM: ", acSVC, " %")
st.sidebar.write("Naive bayes: ", acGNB, " %")
st.sidebar.write("Decision trees: ", acDT, " %")
st.sidebar.write("Random forest: ", acRF, " %")

options = {
    "LR": "Logistic regression",
    "KN": "Nearest neighbor",
    "SVM": "SVM",
    "SVC": "Kernel SVM",
    "GNB": "Naive bayes",
    "DT": "Decision trees",
    "RF": "Random forest"
}
st.subheader("Select your desired algorithm:")
option_selected = st.selectbox("", list(options.items()), 0 , format_func=lambda o: o[1])
st.sidebar.subheader("Input values from medical report:")



def get_input():
    radius_mean = st.sidebar.number_input("Radius mean:", 0.0, 35.0, 6.0, format="%.8f")
    texture_mean = st.sidebar.number_input("Texture mean:", 0.0, 50.0, 20.0, format="%.8f")
    perimeter_mean = st.sidebar.number_input("Perimeter mean:", 0.0, 250.0, 100.0, format="%.8f")
    area_mean = st.sidebar.number_input("Area mean:", 0.0, 3000.0, 500.0, format="%.8f")
    smoothness_mean = st.sidebar.number_input("Smoothness mean:", 0.0, 1.0, 0.1, format="%.8f")
    compactness_mean = st.sidebar.number_input("Compactness mean:", 0.0, 1.0, 0.1, format="%.8f")
    concavity_mean = st.sidebar.number_input("Concavity mean:", 0.0, 1.0, 0.1, format="%.8f")
    concave_points_mean = st.sidebar.number_input("Concave points mean:", 0.0, 1.0, 0.05, format="%.8f")
    symetry_mean = st.sidebar.number_input("Symetry mean:", 0.0, 1.0, 0.2, format="%.8f")
    fractal_dimension_mean = st.sidebar.number_input("Fractal dimension mean:", 0.0, 1.0, 0.05, format="%.8f")
    radius_se = st.sidebar.number_input("Radius se:", 0.0, 10.0, 0.5, format="%.8f")
    texture_se = st.sidebar.number_input("Texture se:", 0.0, 10.0, 5.0, format="%.8f")
    perimeter_se = st.sidebar.number_input("Perimeter se:", 0.0, 50.0, 3.0, format="%.8f")
    area_se = st.sidebar.number_input("Area se", 0.0, 800.0, 50.0, format="%.8f")
    smoothness_se = st.sidebar.number_input("Smoothness se:", 0.0, 1.0, 0.007, format="%.8f")
    compactness_se = st.sidebar.number_input("Compactness se:", 0.0, 1.0, 0.02, format="%.8f")
    concavity_se = st.sidebar.number_input("Concavity se:", 0.0, 1.0, 0.03, format="%.8f")
    concave_points_se = st.sidebar.number_input("Concave points se:", 0.0, 1.0, 0.01, format="%.8f")
    symetry_se = st.sidebar.number_input("Symetry se:", 0.0, 1.0, 0.02, format="%.8f")
    fractal_dimension_se = st.sidebar.number_input("Fractal dimension se:", 0.0, 1.0, 0.003, format="%.8f")
    radius_worst = st.sidebar.number_input("Radius worst:", 0.0, 100.0, 16.0, format="%.8f")
    texture_worst = st.sidebar.number_input("Texture worst:", 0.0, 100.0, 25.0, format="%.8f")
    perimeter_worst = st.sidebar.number_input("Perimeter worst:", 0.0, 500.0, 100.0, format="%.8f")
    area_worst = st.sidebar.number_input("Area worst:", 0.0, 8000.0, 700.0, format="%.8f")
    smoothness_worst = st.sidebar.number_input("Smoothness worst:", 0.0, 1.0, 0.15, format="%.8f")
    compactness_worst = st.sidebar.number_input("Compactness worst:", 0.0, 5.0, 0.25, format="%.8f")
    concavity_worst = st.sidebar.number_input("Concavity worst:", 0.0, 5.0, 0.25, format="%.8f")
    concave_points_worst = st.sidebar.number_input("Concave points worst", 0.0, 1.0, 0.1, format="%.8f")
    symetry_worst = st.sidebar.number_input("Symetry worst:", 0.0, 2.0, 0.33, format="%.8f")
    fractal_dimension_worst = st.sidebar.number_input("Fractal dimension worst", 0.0, 2.0, 0.08, format="%.8f")


    
    

    user_data = {
        "radius_mean": radius_mean,
        "texture_mean": texture_mean,
        "perimeter_mean": perimeter_mean,
        "area_mean": area_mean,
        "smoothness_mean": smoothness_mean,
        "compactness_mean": compactness_mean,
        "concavity_mean": concavity_mean,
        "concave points_mean": concave_points_mean,
        "symetry_mean": symetry_mean,
        "fractal_dimension_mean": fractal_dimension_mean,
        "radius_se": radius_se,
        "texture_se": texture_se,
        "perimeter_se": perimeter_se,
        "area_se": area_se,
        "smoothness_se": smoothness_se,
        "compactness_se": compactness_se,
        "concavity_se": concavity_se,
        "concave points_se": concave_points_se,
        "symetry_se": symetry_se,
        "fractal_dimension_se": fractal_dimension_se,
        "radius_worst": radius_worst,
        "texture_worst": texture_worst,
        "perimeter_worst": perimeter_worst,
        "area_worst": area_worst,
        "smoothness_worst": smoothness_worst,
        "compactness_worst": compactness_worst,
        "concavity_worst": concavity_worst,
        "concave points_worst": concave_points_worst,
        "symetry_worst": symetry_worst,
        "fractal_dimension_worst": fractal_dimension_worst,
        
    }

    #transform user_data into pandas dataframe
    features = pd.DataFrame(user_data, index=[0])
    featuresx = features
    features = sc.transform(features)
    return features, featuresx


user_input, featuresxx = get_input()


predLR = classifierLR.predict(user_input)
predKN = classifierKN.predict(user_input)
predSVM = classifierSVM.predict(user_input)
predSVC = classifierSVC.predict(user_input)
predGNB = classifierGNB.predict(user_input)
predDT = classifierDT.predict(user_input)
predRF = classifierRF.predict(user_input)


#Malignant = 1
#Benign = 0


if st.button("Check results") == True:
    if option_selected[0] == "LR":
        if predLR[0]==1:
            st.subheader("Tumor is most likely malignant.")
            
        elif predLR[0]==0:
            st.subheader("Tumor is most likely benign.")
    elif option_selected[0] == "KN":
        if predKN[0]==1:
            st.subheader("Tumor is most likely malignant.")
        elif predKN[0]==0:
            st.subheader("Tumor is most likely benign.")
    elif option_selected[0] == "SVM":
        if predSVM[0]==1:
            st.subheader("Tumor is most likely malignant.")
        elif predSVM[0]==0:
            st.subheader("Tumor is most likely benign.")
    elif option_selected[0] == "SVC":
        if predSVC[0]==1:
            st.subheader("Tumor is most likely malignant.")
        elif predSVC[0]==0:
            st.subheader("Tumor is most likely benign.")
    elif option_selected[0] == "GNB":
        if predGNB[0]==1:
            st.subheader("Tumor is most likely malignant.")
        elif predGNB[0]==0:
            st.subheader("Tumor is most likely benign.")
    elif option_selected[0] == "DT":
        if predDT[0]==1:
            st.subheader("Tumor is most likely malignant.")
        elif predDT[0]==0:
            st.subheader("Tumor is most likely benign.")
    elif option_selected[0] == "RF":
        if predRF[0]==1:
            st.subheader("Tumor is most likely malignant.")
        elif predRF[0]==0:
            st.subheader("Tumor is most likely benign.")


#Add graph: Malignant mean, Benign mean values + Input

#featuresxx are values from input before transform

#Check means!
data_M = b_data[b_data["diagnosis"] == "M"]
data_B = b_data[b_data["diagnosis"] == "B"]

describe_M = data_M.describe()
describe_B = data_B.describe()


mean_B = describe_B.iloc[1, :]
mean_M = describe_M.iloc[1, :]

mean_B = mean_B[1:31]
mean_M = mean_M[1:31]



mean_B.name = "Benign means"
#mean_B = mean_B.rename_axis("Benign")
#mean_M = mean_M.rename_axis("Malign")
mean_M.name = "Malignant means"
#RENAME ONE PANDA SERIES NAME

mean_B=(mean_B-mean_B.mean())/mean_B.std() #normalize
mean_M=(mean_M-mean_M.mean())/mean_M.std() #normalize

mean_frame = pd.merge(mean_B, mean_M, right_index = True, 
               left_index = True)

if st.button("Display graphical representation") == True:
    featuresxx = featuresxx.transpose()
    featuresxx = ((featuresxx - featuresxx.mean())/featuresxx.std())
    mean_frame = pd.merge(mean_frame, featuresxx, right_index = True, 
                left_index = True)
    mean_frame.columns = ["Benign mean values", "Malignant mean values", "User input values"]
    mean_frame["Benign mean values"] = mean_frame["Benign mean values"] + 1 
    mean_frame["Malignant mean values"] = mean_frame["Malignant mean values"] + 1
    mean_frame["User input values"] = mean_frame["User input values"] + 1

    print(mean_frame)
    #WORKS:!
    colors = ["#eb75c7", "#e64c29", "#5798fa"]
    st.subheader("Representation of normalized input values and mean normalized values of malignant and benign tumors from medical records:")
    fig, ax = plt.subplots()
    ax = mean_frame.plot.bar(rot=0, width=0.7, color=colors, figsize=(12,10))
    plt.xticks(rotation=90)
    plt.ylabel("Normalized values")
    ax.set_yticks([])
    st.pyplot()

    colors2 = ["#eb75c7", "#e64c29", "#22470b"]
    st.subheader("Kernel density estimation:")
    fig, ax = plt.subplots()
    ax = mean_frame.plot.kde(rot=0, color=colors, figsize=(12,10))
    plt.xticks(rotation=90)
    ax.set_yticks([])
    st.pyplot()

print(X)