import numpy as np, matplotlib.pyplot as plt, pandas as pd

dataset = pd.read_csv('Data.csv')
print(dataset.info())
print()

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
print('Features :\n',x)
print('Target :\n',y)
print()

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

x[:, 1:3] = imputer.fit_transform(x[:, 1:3])
print('After Imputing :\n',x)
print()

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder = 'passthrough')
x = pd.DataFrame(ct.fit_transform(x))
print('One Hot Encoding :\n', x)
print()

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y = le.fit_transform(y)
print('Label Encoder :\n ', y)
print()

#Feature Scaling -age and salary(trying to scale down as sys will get affected with huge variation in the values)
#StandardScaler = (x_mean)/stdev (values will range from -1 to 1 mean to be 0)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size= 0.2, random_state = 1)
print('X_Train : \n',x_train)
print('X_Test : \n',x_test)
print()

print('Y_Train : \n',y_train)
print('Y_Test : \n',y_test)
print()

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train.iloc[:, 1:3] = sc.fit_transform(x_train.iloc[:, 1:3])
x_test.iloc[:, 1:3] = sc.transform(x_test.iloc[:, 1:3])

print('After scaling the values from -1 to 1 : \n')
print(x_train)
print(x_test)
