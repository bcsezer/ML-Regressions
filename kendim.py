#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 13:09:06 2019

@author: cemsezeroglu
"""

#kütüphanelerin eklenmesi

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#verinin yüklenmesi

veriler = pd.read_csv("odev_tenis.csv")
print(veriler)

#Eksik veriler

#eksik veri olmadığından bu kısımda bir işlem yok.

#Verilerin Kategorikten - Sayısal verilere dönüşmesi. yani numeric
#gerekli verilere teker teker label encoder işlemi yapılacağına map ile tek satırda yapılıyor

veriler2 = veriler.apply(LabelEncoder().fit_transform)


#Burada OneHotEncoder yaptık türleri 0 1 olarak ayırdık 3 tane seçenek vardı 1-0 oldu.
outlook = veriler2.iloc[:,:1].values
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features="all")
outlook = ohe.fit_transform(outlook).toarray()
print(outlook)

#sayısal işlemlerini gerçekleştirdiğimiz verileri tek bir Dataframe de toplama işlemleri
havadurumu = pd.DataFrame(data = outlook, index=range(14),columns = ["overcast","sunny","rainy"])
sonveriler = pd.concat([havadurumu,veriler.iloc[:,1:3]],axis = 1)
sonveriler = pd.concat([veriler2.iloc[:,-2:],sonveriler],axis = 1)
#Veriler numeric hale geldi algoritma artık verileri anlayabilir.

#verilerin egitim ve test icin bolunmesi

#hoca örnekte train test split kullnarak yapıyordu bende öyle yapıcam ama başka yöntemler var.

from sklearn.cross_validation import train_test_split


#humidity yi tahmin edicez istediğimizi tahmin edebiliriz bu yüzden humidity bağımlı değişken
#yani denklemde Y .ilk başta bağımsız değişkenleri yazıyoruz fonksiyona.
x_train,x_test,y_train,y_test = train_test_split(sonveriler.iloc[:,:-1],sonveriler.iloc[:,-1:],test_size = 0.33,random_state = 0)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)

print(y_pred)

#backward elimination
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((14,1)).astype(int),values = sonveriler.iloc[:,:-1],axis = 1)
X_l = sonveriler.iloc[:,[0,1,2,3,4,5]].values
r_ols = sm.OLS(endog = sonveriler.iloc[:,-1:], exog = X_l)

r = r_ols.fit()
print(r.summary())

sonveriler = sonveriler.iloc[:,1:]

import statsmodels.formula.api as sm
X = np.append(arr = np.ones((14,1)).astype(int),values = sonveriler.iloc[:,:-1],axis = 1)
X_l = sonveriler.iloc[:,[0,1,2,3,4,]].values
r_ols = sm.OLS(endog = sonveriler.iloc[:,-1:], exog = X_l)

r = r_ols.fit()
print(r.summary())

x_train = x_train.iloc[:,1:]
x_test = x_test.iloc[:,1:]

regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)



















