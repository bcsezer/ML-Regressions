#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 14:13:10 2019

@author: cemsezeroglu
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

veriler_son = pd.read_csv("maaslar.csv")


x = veriler_son.iloc[:,1:2] #egitim seviyesi
y = veriler_son.iloc[:,2:]#maas

from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor(random_state = 0,n_estimators = 10)
#n_estimators kaç tane decision tree çizileceğini söylüyoruz

rf_reg.fit(x,y)



print("random forest tahmin")
print(rf_reg.predict(11))

plt.scatter(x,y,color = "red")

plt.plot(x,rf_reg.predict(x),color="blue")
plt.show()


from sklearn.metrics import r2_score
print("random forest R2 değeri : ")
print(r2_score(y,rf_reg.predict(x)))