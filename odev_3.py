#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 12:42:56 2019

@author: cemsezeroglu
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score
import statsmodels.api as sm

#veri yüklenmesi
odev3_data = pd.read_csv("maaslar_yeni.csv")

#Maaşı bulmak istiyoruz.

# BU ESKİSİ x_bagımsız_degisken = odev3_data.iloc[:,2:5].values
x_bagımsız_degisken = odev3_data.iloc[:,2:3].values
#3parameter 
#x i böyle seçtik çünkü p value su düşük olanı aldık.
y_bagımlı_degisken = odev3_data.iloc[:,5:].values

#simdi regresyon modellerini deneyip p value bakıcaz

#Linear Regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg=lin_reg.fit(x_bagımsız_degisken,y_bagımlı_degisken)

#R values, r2 values check
print("Linear Regression P")
model_lin = sm.OLS(lin_reg.predict(x_bagımsız_degisken),x_bagımsız_degisken)
print(model_lin.fit().summary())

'''
x1             0.000  -->p values  
x2             0.997  -->p values  
x3             0.437  -->p values  
    yani 0.05 olarak kabul ettiğimiz sınırda 2 ve 3 çok yüksek onları eliyoruz.
'''    
#Polynomial Regression

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
x_poly = poly_reg.fit_transform(x_bagımsız_degisken)
lin_reg_new = LinearRegression()
lin_reg_new.fit(x_poly,y_bagımlı_degisken)

print("Polynomial Regression p ")
model_pol = sm.OLS(lin_reg.predict(x_bagımsız_degisken),x_bagımsız_degisken)
print(model_pol.fit().summary())

#Suppor Vector Regression
from sklearn.preprocessing import StandardScaler
stndrd = StandardScaler()
x_standard = stndrd.fit_transform(x_bagımsız_degisken)
stndrd2 = StandardScaler()
y_standard = stndrd2.fit_transform(y_bagımlı_degisken)
from sklearn.svm import SVR
svr_reg = SVR(kernel = "rbf")
svr_reg.fit(x_standard,y_standard)
print(" Support Vector Regression p ")
model_svr = sm.OLS(svr_reg.predict(x_standard),x_standard)
print(model_svr.fit().summary())

#Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor

r_dt = DecisionTreeRegressor(random_state = 0)

r_dt.fit(x_bagımsız_degisken,y_bagımlı_degisken)
print("Decision Tree Regression p ")
model_dt = sm.OLS(r_dt.predict(x_bagımsız_degisken),x_bagımsız_degisken)
print(model_dt.fit().summary())

#Random Forest Regression
from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor(random_state = 0,n_estimators = 10)
#n_estimators kaç tane decision tree çizileceğini söylüyoruz

rf_reg.fit(x_bagımsız_degisken,y_bagımlı_degisken)
print("Random Forest Tree Regression p ")
model_rf = sm.OLS(rf_reg.predict(x_bagımsız_degisken),x_bagımsız_degisken)
print(model_rf.fit().summary())



#Tahminler
print("---------------------------")

print("Linear Regression Tahmin : ")
print(lin_reg.predict(11))#[[34716.66666667]]
print(lin_reg.predict(6.6))#[[16923.33333333]]
print("---------------------------")

print("Polynomial Regression Tahmin : ")
print(lin_reg_new.predict(poly_reg.fit_transform(11))) #[[89041.66666665]]
print(lin_reg_new.predict(poly_reg.fit_transform(6)))
print("---------------------------")

print("Support Vector Regression Tahmin : ")
print(svr_reg.predict(11))
print(svr_reg.predict(6.6))
print("---------------------------")

print("decision tree tahmin : ")

print(r_dt.predict(50))
print(r_dt.predict(6.6))
print("---------------------------")

print("random forest tahmin")
print(rf_reg.predict(11))

print(" Tahminler Yukarıda ")






#Tek parametreleri olarak R2 sonuçları : 
'''
Linear Regression 
R-squared:                       0.942

Polynomial Regression 
R-squared:                       0.942

Support Vector Regression
R-squared:                       0.770

Decision tree 
R-squared:                       0.751

Random forest
R-squared:                       0.719
'''