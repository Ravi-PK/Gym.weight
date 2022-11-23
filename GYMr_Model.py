    # -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 22:16:05 2022

@author: Ravi 1
"""

import pandas as pd
import numpy as np

cd

import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

cd

cd Downloads

gym = pd.read_excel('dataGYM.xlsx')

gym.head()

gym['Class']=LabelEncoder().fit_transform(gym['Class'])

gym.head()

gym['Class'].value_counts()

X=gym.iloc[:,:5]
X.head()

y = gym.iloc[:,5:]
y.head()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)
model_GYM = RandomForestClassifier(n_estimators=20)
model_GYM.fit(X_train, y_train)

print(model_GYM)


#Make predictions
expected = y_test
predicted = model_GYM.predict(X_test)

metrics.classification_report(expected, predicted)
metrics.confusion_matrix(expected, predicted)

import pickle

pickle.dump(model_GYM, open("Model_GYM.pkl","wb"))

model = pickle.load(open("Model_GYM.pkl","rb"))








