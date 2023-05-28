import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df= pd.read_csv("D:\ALL FILES\Desktop\heart.csv")
#print(df.duplicated().sum())
#df=df.drop_duplicates()
#df.info()
#print(df.head())
df = pd.get_dummies(df,columns =["cp","restecg","slp","thall"])
lc = df.pop('output')
df.insert(23,'output',lc)
#print(df.head())


x=df.iloc[:,:23]
x_train,x_test,y_train,y_test=train_test_split(x,df.output,test_size=0.2,random_state=42)

lr = LogisticRegression(max_iter=15000,solver='sag')
lr.fit(x_train,y_train)
pickle.dump(lr,open('model.pkl','wb'))
model= pickle.load(open('model.pkl','rb'))

#print(lr.score(x_test,y_test))
#print(lr.predict(x_test))
#print(lr.predict_proba(x_test))