"""
Created on Thu Sep  19 19:53:08 2020

@author: emman
"""

# Exploratory data analysis packages 
import pandas as pd
import numpy as np

# Visualization packages
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import plotly.express as px

# Streamlit componets for automation
import streamlit as st
import streamlit.components.v1 as components

# Machine Learning packages
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score 
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.cluster import KMeans

# Function to contain all the functionalities of the app
def main():
    
    # Gives the title of the APp
    st.title("Employee Churn Predictor")
    

    # Core functionalities of the App
    functions=['Home', 'Data Report', 'Visualize_dataset', 'Predictions']
    choice=st.sidebar.selectbox("Select choice" ,functions )
    

    if choice=='Home':
        html_temp="""
        <img src="https://user-images.githubusercontent.com/68768460/93686845-a2bd8300-faa8-11ea-8114-1b9669d26c38.jpg" width= "500", height="100" >
        """
        components.html(html_temp)    

    elif choice=='Data Report':
        def load_data():
            df=pd.read_csv("employee_data.csv")
            return df
        data=pd.DataFrame(load_data())
        st.write(data.head())

        st.write("Rows&Columns:", data.shape)
        st.write("Descriptive Statistics:",data.describe())
        st.write("Nnull Values:", data.isna().sum())
        st.write("Data Types:", data.dtypes)

            
    
    elif choice=='Visualize_dataset':
        def load_data():
            df=pd.read_csv("employee_data.csv")
            return df
        data=pd.DataFrame(load_data())
        st.write(data.head())

        
        if st.sidebar.button("Bar Chart"):
            features=['quit','number_project', 'time_spend_company', 'Work_accident', 'promotion_last_5years', 'department', 'salary']
            fig=plt.subplots(figsize=(10,15))
            for i, j in enumerate(features):
                st.write(plt.subplot(4, 2, i+1))
                st.write(plt.subplots_adjust(hspace=1.0))
                st.write(sns.countplot(x=j, data=data))
                st.write(plt.xticks(rotation=90))
                st.write(plt.ylabel('Number of employees'))
                st.pyplot()

        if st.sidebar.button("Turnover Frequency on other variables"):
            bar1=pd.crosstab(data.salary, data.quit).plot(kind='bar')
            st.write(bar1)
            st.subheader("Turnover Frequency based on Salary levels")
            st.pyplot()



            bar2=pd.crosstab(data.department, data.quit).plot(kind='bar')
            st.write(bar2)
            st.subheader("Turnover Frequency on department")
            st.pyplot()

            
            bar3=pd.crosstab(data.Work_accident, data.quit).plot(kind='bar')
            st.write(bar3)
            st.subheader("Turnover Frequency based on work accident")
            st.pyplot()

               
            bar4=pd.crosstab(data.promotion_last_5years, data.quit).plot(kind='bar')
            st.write(bar3)
            st.subheader("Turnover Frequency based on Promotion")
            st.pyplot()






        if st.sidebar.button("Pie Chart"):
            fig=plt.figure(figsize=(5,6))

            plt.subplot(2,2,1)
            st.subheader("Pie Chart Showing the number of empoloyees who quit and those who stayed")
            pie_chart=data['quit'].value_counts().plot.pie(autopct='%1.1f%%')
            st.write(pie_chart)
            st.pyplot()
                
            plt.subplot(2,2,2)
            st.subheader("Pie Chart showing the salary levels of employees")
            pie_chart=data['salary'].value_counts().plot.pie(autopct='%1.1f%%')
            st.write(pie_chart)
            st.pyplot()
                
            plt.subplot(2,2,3)
            st.subheader("Pie Chart showing the various departments in the firm")
            pie_chart=data['department'].value_counts().plot.pie(autopct='%1.1f%%')
            st.write(pie_chart)
            st.pyplot()
                
            plt.subplot(2,2,4)
            st.subheader("Pie Chart showing the percentage of empolyees who had promotion for the past 5yrs")
            pie_chart=data['promotion_last_5years'].value_counts().plot.pie(autopct='%1.1f%%')
            st.write(pie_chart)
            st.pyplot()
                
            st.pyplot()

        if st.sidebar.button("Correlation Matrix"):
            st.subheader("Correlation Matrix Visualization")
            st.write(sns.heatmap(data.corr(), annot=True))
            st.pyplot()

            
        if st.sidebar.button("Visualize cluster of employees who quit"):
            st.subheader("Cluster of Employees who quit")
            # filter Data
            left_emp=data[['satisfaction_level', 'last_evaluation']][data.quit==1]

            #create groups using Kmeans clustering
            kmeans=KMeans(n_clusters=3, random_state=0).fit(left_emp)

            #Add new column 'label' and assign cluster labels
            left_emp['label']=kmeans.labels_

            #Draw scatter plot
            st.write(plt.scatter(left_emp['satisfaction_level'], left_emp['last_evaluation'], c=left_emp['label'], cmap='Accent'))
            st.write(plt.xlabel('Satisfaction Level'))
            st.write(plt.ylabel('Last Evaluation'))
            st.write(plt.title('Cluster of Employees who left'))
            st.write(plt.show())
            st.pyplot()


            st.pyplot()


    elif choice=='Predictions':
        st.subheader("Select type of Prediction MOdel")
        options=['Prediction Model','Logistic Regression' ,'Decision Tree','Random forest', 'GradientBoosting']
        choice=st.sidebar.selectbox("Select ML models", options)
    
    # Defining a preprocessor function to prepare the dataset for model building
        def load_data():
            df=pd.read_csv("employee_data.csv")
            encoder=LabelEncoder()
            for col in df.columns:
                df[col]=encoder.fit_transform(df[col])
            return df
        data=pd.DataFrame(load_data())
            
   

        x=data.loc[:, data.columns !='quit']
        y=data['quit']

        # we split the dataset into training and testing 
        def split(df):
            x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=st.number_input("enter test size", 0.1), random_state=st.number_input("enter randon state",min_value=1, max_value=None), stratify=y)
            return x_train, x_test, y_train, y_test
                
        x_train, x_test, y_train, y_test=split(data)

        if choice=='Logistic Regression':
            st.sidebar.subheader('Model hyperparameters')
            c=st.sidebar.number_input('Regularization parameter', 0.01, 10.0, step=0.01)

            st.subheader("Logistic Regression Results:")
            model =LogisticRegression()
            model.fit(x_train, y_train)
            accuracy=model.score(x_test, y_test)
            y_pred=model.predict(x_test)
            st.write("Accuracy:", accuracy)
            st.write("Precision:", precision_score(y_test, y_pred))
            st.write("Recall:", recall_score(y_test, y_pred))

                
            if st.sidebar.checkbox("Confusion Matrix"):
                st.subheader("confusion Matrix")
                plot_confusion_matrix(model, x_test, y_test)
                st.pyplot()
                    
            if st.sidebar.checkbox("ROC Curve"):
                st.subheader("ROC Curve")
                plot_roc_curve(model, x_test, y_test)
                st.pyplot()
            if st.sidebar.checkbox("Precision-Recall Curve"):
                st.subheader("Precision-Recall Curve")
                plot_precision_recall_curve(model, x_test, y_test)
                st.pyplot()

            if st.sidebar.checkbox('Predict whether an employee will stay or Not'):
               prediction= model.predict([[st.number_input('Satisfaction Level'), st.number_input('last evaluation'), st.number_input('number of prjects'), st.number_input('average monthly hours'), st.number_input('time spend company'),st.number_input('work accident'), st.number_input('promotion last 5years'), st.number_input('Department'), st.number_input('salary') ]])
               st.write('Your predicted outcome:',prediction )

        if choice=='Random forest':
            st.sidebar.subheader('Model hyperparameters')
            n_estimators=st.sidebar.number_input('the number of trees in the forest', 100, 5000, step=10)
            max_depth=st.sidebar.number_input('the maximum depth of tree', 1, 20, step=1)
            criterion=st.sidebar.radio('Bootsrap samples when building trees',('gini', 'entropy'))


            st.subheader(" Random Forest Results:")
            model =RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,criterion=criterion)
            model.fit(x_train, y_train)
            accuracy=model.score(x_test, y_test)
            y_pred=model.predict(x_test)
            st.write("Accuracy:", accuracy)
            st.write("Precision:", precision_score(y_test, y_pred))
            st.write("Recall:", recall_score(y_test, y_pred))

            if st.sidebar.checkbox("Confusion Matrix"):
                st.subheader("confusion Matrix")
                plot_confusion_matrix(model, x_test, y_test)
                st.pyplot()
                    
            if st.sidebar.checkbox("ROC Curve"):
                st.subheader("ROC Curve")
                plot_roc_curve(model, x_test, y_test)
                st.pyplot()
            if st.sidebar.checkbox("Precision-Recall Curve"):
                st.subheader("Precision-Recall Curve")
                plot_precision_recall_curve(model, x_test, y_test)
                st.pyplot()
         

        if choice=='Decision Tree':
            st.sidebar.subheader('Model hyperparameters')
            max_depth=st.sidebar.number_input('the maximum depth of tree', 1, 20, step=1)
            criterion=st.sidebar.radio('Bootsrap samples when building trees',('gini', 'entropy'))
            splitter=st.sidebar.radio('Bootsrap samples when building trees',('best', 'random'))


            st.subheader(" Decision Tree Results:")
            model =DecisionTreeClassifier(criterion=criterion, splitter= splitter, max_depth=max_depth)
            model.fit(x_train, y_train)
            accuracy=model.score(x_test, y_test)
            y_pred=model.predict(x_test)
            st.write("Accuracy:", accuracy)
            st.write("Precision:", precision_score(y_test, y_pred))
            st.write("Recall:", recall_score(y_test, y_pred))

                
            if st.sidebar.checkbox("Confusion Matrix"):
                st.subheader("confusion Matrix")
                plot_confusion_matrix(model, x_test, y_test)
                st.pyplot()
                    
            if st.sidebar.checkbox("ROC Curve"):
                st.subheader("ROC Curve")
                plot_roc_curve(model, x_test, y_test)
                st.pyplot()
            if st.sidebar.checkbox("Precision-Recall Curve"):
                st.subheader("Precision-Recall Curve")
                plot_precision_recall_curve(model, x_test, y_test)
                st.pyplot()

            if choice=="Visualize Decision Tree":
                st.subheader("Decision Tree Visualization")
                plt.figure(figsize=(10,10))
                tree.plot_tree(model, filled=True)
                st.pyplot()

        
    

import warnings

def fxn():
    warnings.warn("deprecated", DeprecationWarning)
    
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
fxn()
st.set_option('deprecation.showPyplotGlobalUse', False)

     
       
st.set_option('deprecation.showfileUploaderEncoding', False)
if __name__ == "__main__":
  main()

