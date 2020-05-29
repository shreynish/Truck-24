import pandas as pd

dataset=pd.read_csv('project.csv')

x=dataset.iloc[:,0:4]
y=dataset.iloc[:,-1]

from sklearn.linear_model import LinearRegression
model = LinearRegression()

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=1/3,random_state=20)

model.fit(X_train,Y_train)

model.predict(X_test)

model.predict([[100,1000,1200,800]])
