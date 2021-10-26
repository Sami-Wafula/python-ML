#importing modules

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# preparing the dataset for easy fitting of the Linear Regression model
def prepare_data(df, forecast_col, forecast_out, test_size):
	label = df[forecast_col].shift(-forecast_out)
	X = np.array(df[[forecast_col]])
	X = preprocessing.scale(X)
	X_lately = X[-forecast_out:]
	X = X[:-forecast_out]
	label.dropna(inplace=True)
	y = np.array(label)
	X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=test_size, random_state=0)
	
	response = [X_train, X_test, Y_train, Y_test, X_lately]
	return response
	
#reading the data
df = pd.read_csv('C:\\Users\\Sami\\Downloads\\TTE.csv')
print (df)

#declaring variables
forecast_col = 'Close'
forecast_out = 5
test_size = 0.2

#splitting the data and fitting it into the regression model
X_train, X_test, Y_train, Y_test, X_lately = prepare_data(df, forecast_col, forecast_out, test_size);
learner = LinearRegression ()

learner.fit(X_train, Y_train)

#predicting the output
score = learner.score(X_test, Y_test)
forecast = learner.predict(X_lately)
response={}
response['test_score']=score
response['forecast_set']=forecast

print(response)