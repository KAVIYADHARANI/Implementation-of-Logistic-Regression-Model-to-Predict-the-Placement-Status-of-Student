# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:

1.Import the standard libraries such as pandas module to read the corresponding csv file.

2.Upload the dataset values and check for any null or duplicated values using .isnull() and .duplicated() function respectively.

3.Import LabelEncoder and encode the corresponding dataset values.

4.Import LogisticRegression from sklearn and apply the model on the dataset using train and test values of x and y and Predict the values of array using the variable y_pred.

5.Calculate the accuracy, confusion and the classification report by importing the required modules such as accuracy_score, confusion_matrix and the classification_report from sklearn.metrics module.

6.Apply new unknown values and print all the acqirred values for accuracy, confusion and the classification report. 

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Kaviya D
RegisterNumber:  212223040089
*/
```
```
import pandas as pd
data=pd.read_csv('Placement_Data.csv')
data.head()


data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis = 1)#removes the specified row or column
data1.head()


data1.isnull().sum()


data1.duplicated().sum()


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1


x=data1.iloc[:,:-1]
x


y=data1["status"]
y


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size = 0.2,random_state = 0)


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear")# a library for large
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred
```
## Output:
![367200450-9d94a240-b858-4827-8fab-fc955ffa41ce](https://github.com/user-attachments/assets/d51b9d5d-66ac-4d04-8a75-139d545015d5)
![367199861-2fc0657f-732b-43ac-b3e8-09c3c387ae81](https://github.com/user-attachments/assets/b91cbe6f-c941-415e-852e-41a5cb7513e7)
![367199902-cc62fca3-573b-4edd-abd2-dd75321e407b](https://github.com/user-attachments/assets/25f7ed9e-29d2-4a02-bac7-594be812f180)

![367199993-34d28793-3c42-4430-8cf7-f46bd66681ab](https://github.com/user-attachments/assets/5c2b995f-1f73-41bc-97a1-13323b22e03b)
![367200065-d5f13a72-cf96-48de-8f1d-36d36c0b7a6f](https://github.com/user-attachments/assets/0c55afb1-4af0-440a-9637-a67be2b688c6)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
