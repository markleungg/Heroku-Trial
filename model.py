import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.datasets import load_boston

boston = load_boston()

df_boston = pd.DataFrame(data= boston.data, columns=boston["feature_names"]) # We have the list of the column names(feature_name). So, we can change the column names.
df_boston

df_boston["MEDV"] = boston.target

X = df_boston[["LSTAT","RM","PTRATIO"]]
y = df_boston["MEDV"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25, random_state=101)

from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train, y_train)
#save model to disk
pickle.dump(lm, open('model.pkl', 'wb'))

#loading model to compare results
model = pickle.load(open('model.pkl', 'rb'))
print(model.predict([[2.94, 6.998, 18.7]]))