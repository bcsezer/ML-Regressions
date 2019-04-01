#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 13:06:09 2019

@author: cemsezeroglu
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("maaslar.csv")



x_egitim = data.iloc[:,1:2].values
y_maas = data.iloc[:,2:].values


from sklearn.preprocessing import StandardScaler

stndrd = StandardScaler()
x_standard = stndrd.fit_transform(x_egitim)

stndrd2 = StandardScaler()
y_standard = stndrd2.fit_transform(y_maas)

from sklearn.svm import SVR

svr_reg = SVR(kernel = "rbf")
svr_reg.fit(x_standard,y_standard)

plt.scatter(x_standard,y_standard,color = "red")
plt.plot(x_standard,svr_reg.predict(x_standard),color = "blue")
plt.show()

print("svm değerleri")
print(svr_reg.predict(11))
print(svr_reg.predict(6.6))

from sklearn.metrics import r2_score
print("Support Vector Regression R2 değeri : ")
print(r2_score(y_standard,svr_reg.predict(x_standard)))