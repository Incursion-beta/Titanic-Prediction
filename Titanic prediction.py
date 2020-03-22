# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 14:45:09 2020

@author: nitesh vamshi
"""

import numpy as np
import pandas as pd
train = pd.read_csv('train.csv')
test=pd.read_csv('test.csv') 
train = train.drop(['PassengerId','Name','Ticket','Cabin','Embarked'], axis=1)
test = test.drop(['Name','Ticket','Cabin','Embarked'], axis=1)
train['Age'].fillna(train['Age'].mean(),inplace=True)
test['Age'].fillna(test['Age'].mean(),inplace=True)
test['Fare'].fillna(test['Fare'].mean(),inplace=True)
train['Sex'] = train['Sex'].replace(['female','male'],[0,1])
test['Sex'] = test['Sex'].replace(['female','male'],[0,1])
X_train = train.drop("Survived", axis=1)
Y_train = train["Survived"]
X_test  = test.drop("PassengerId", axis=1).copy()
from sklearn.ensemble import RandomForestClassifier as RandomForest
model = RandomForest(n_estimators=2000,bootstrap=True)
model.fit(X_train,Y_train)
result = model.predict(X_test)
submission = pd.DataFrame({"PassengerId": test["PassengerId"],"Survived": result})
submission.to_csv("submission.csv", index=False)





