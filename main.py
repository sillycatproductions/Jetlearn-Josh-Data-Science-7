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