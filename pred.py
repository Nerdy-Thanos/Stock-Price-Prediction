#importing the required packages/libraries
import pandas as pd
import numpy as np
import quandl
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import seaborn as sns

df = pd.read_csv("new_book.csv")
#filling null values
df['Adj Close'].fillna(df['Adj Close'].mean())

#df.head()

#Create a new df for manipulation/adding/removing coloumns
new_df = df[['Adj Close']]
#new_df.head()

#variable to predict 'n' days into future
forecast_out  = 10
#add coloumn with target/dependent varible shifted by 'n' units
new_df['Prediction'] = new_df['Adj Close'].shift(-forecast_out)
#new_df.head()

### Create independent dataset X ###
#convert dataset to numpy array

X = np.array(new_df.drop(['Prediction'],1))

#Removing the last 'n' rows

X = X[:-forecast_out]

### Create Dependent dataset Y ###
#Convert dataset to numpy array

Y = np.array(new_df['Prediction'])
Y = Y[:-forecast_out]

#split the data into 80% train and 20% test
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size = 0.2)

#Create SVM
svr_rbf = SVR(kernel = 'rbf', C = 1e3, gamma = 0.1)
svr_rbf.fit(x_train,y_train)

#testing accuracy of the SVM 
svr_confidence = svr_rbf.score(x_test,y_test)
print("SVM Confidence : ",svr_confidence)

#Create Linear Regression model
lr = LinearRegression()
lr.fit(x_train,y_train)

#testing accuracy of Linear Regression model
lr_confidence = lr.score(x_test,y_test)
print("Linear Regression confidence : ",lr_confidence)

x_forecast = np.array(new_df.drop(['Prediction'],1))[-forecast_out:]
print(x_forecast)
predict = lr.predict(x_forecast)
#print(predict)
svm_pred = svr_rbf.predict(x_forecast)
#print(svm_pred)

#compare predicted values to actual values
actual = df['Adj Close'].tail(forecast_out)
actual = np.array(actual)
final = [actual,svm_pred,predict]
#Creating a sample Dataframe to compare values
pred_df = pd.DataFrame(data=actual,columns=['Actual'])
pred_df['SVM_Pred'] = svm_pred
pred_df['LR_Pred'] = predict
print(pred_df)