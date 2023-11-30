import streamlit as st 
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn import datasets # these dataset contains dummy dataset
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import model_selection
from sklearn.preprocessing import LabelEncoder
matplotlib.use('Agg')


st.subheader("Welcome choice an option at the left")
# Creating functions to handle actions
def actions():
    activities = ['EDA', 'Visualization', 'Prediction']
    choice = st.sidebar.selectbox("Select Activity", activities)

    if choice == "EDA":
        st.subheader("Exploratory Data Analysis")
        data = st.file_uploader("Upload dataset", type=["csv", "xlsx"])
        
        if data is not None:
            st.success("Data uploaded successfully")
            df =pd.read_csv(data)
            st.dataframe(df.head())
            st.write("Rows and columns are:", df.shape)
            st.write("Number of missing column:", df.isna().sum())
            st.write("The information from the file is:", df.info())
            st.write('Description of the file:', df.describe())
            st.write("The correlation of columns is", df.corr())
    elif choice == "Visualization":
        st.subheader("Visualization")
        data = st.file_uploader("Upload dataset", type=["csv", "xlsx"])
        if data is not None:
            st.success("Data uploaded successfully")
            df =pd.read_csv(data)
            st.dataframe(df.head())
            X = df.iloc[:,0:1]
            Y = df.iloc[:, 1:2]
            Z = df.iloc[:, 2:3]
            A = df.iloc[:, 3:4]
            st.write("The graph",sns.countplot(x = X.squeeze()))
            st.pyplot()
            st.set_option('deprecation.showPyplotGlobalUse', False)
    
    # This shows how you can do it for a file without asking the indiviual to choose a file.
    # Drag the file you want to use and place in vscode folder
    elif choice == "Prediction":
            st.subheader("Prediction")
            df = pd.read_csv("heights_weights.csv")
            # st.dataframe(df.head())
            #Dividing my data into X and y variables
            x=df.iloc[:,0:-1]
            y=df.iloc[:,-1]
            x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
            model = KNeighborsClassifier()
            model.fit(x_train,y_train)
            y_predict = model.predict(x_test)
            # st.write("Accuracy Score:", accuracy_score(y_test,y_predict))
            height = st.number_input("Enter your height in cm")
            weight = st.number_input("Enter your weight in lb")
            if st.button("Predict"):
                result = model.predict([[height,weight]])
                if result == 1:
                    st.success("You are a Female")
                else:
                    st.success("You are a Male")
    else:
        st.subheader("Welcome to Tech365")
        st.subheader("Select an option on the left to get started")

actions()