import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

df= pd.read_csv("D:\ALL FILES\Desktop\heart.csv")

#df.duplicated().sum()
#df=df.drop_duplicates()

df = pd.get_dummies(df,columns =["cp","restecg","slp","thall"])
lc = df.pop('output')
df.insert(23,'output',lc)
x=df.iloc[:,:23]

x_train,x_test,y_train,y_test=train_test_split(x,df.output,test_size=0.2,random_state=42)


model1 = SVC(C=.1,kernel='linear',gamma=1).fit(x_train,y_train)
model2=LogisticRegression(max_iter=15000,solver='sag').fit(x_train,y_train)
model3=DecisionTreeClassifier().fit(x_train,y_train)
model4=KNeighborsClassifier(n_neighbors=6).fit(x_train,y_train)
model5=RandomForestClassifier(n_estimators = 100).fit(x_train,y_train)


print("Accuracy of SVM : - ")
print(model1.score(x_test,y_test))
print("Accuracy of Logistic Regression : - ")
print(model2.score(x_test,y_test))
print("Accuracy of Decision Tree : - ")
print(model3.score(x_test,y_test))
print("Accuracy of KNN : - ")
print(model4.score(x_test,y_test))
print("Accuracy of Random Forest : - ")
print(model5.score(x_test,y_test))