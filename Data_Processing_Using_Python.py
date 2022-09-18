import math
import pandas as np

"""Even or Odd Number Testing"""
def EvenOrOdd(integer):
    if integer%2==0:
        return "Even"
    else:
        return "Odd"
print(EvenOrOdd(45))

"""Handling Misssing Data"""
"""replace the empty spaces with the "mean" of other values in that particular column."""
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

"""Encoding Categorical Data"""
"""
we need to have three separate columns for ECS, ENG and MFG and 
put 1 or 0 against them if they are present or absent for a particular row. 
"""
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
"""
we have to encode "Yes" and "No" in "y" variable.
 Since, we have only two categories there,
 we can use 0 or 1 in the same column and need not to have two columns for "Yes" and "No" 
 as we had in the case with "X". So, we use LabelEncoder() for this as shown below.
"""
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

"""Splitting the Dataset"""
"""
When we train our model with dataset, we need to test it afterwards,
 To achieve this, we split our dataset into two sets. 
 First one we use to train our model and the second one we use to test our model
"""
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
"""
We have just a one liner code to achieve this. 
The code is self-explanatory. 
Here, test_size=0.2 means 20%. Consider random_state to be 1 for now.
"""
print(X_train)
print(X_test)
print(y_train)
print(y_test)

"""Feature Scaling"""
"""
-Feature Scaling Technique allow us to put all our features on the same scale.
-So, feature scaling stops some features to be dominated by others in such a way 
that some feature is not even considered by the ML model.
"""
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])
"""
Here we are using "3:" to take columns from index 3 to end. 
Note, this variable contains our latest dataset after categorical encodings.
"""
print(X_train)
print(X_test)


