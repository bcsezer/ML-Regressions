#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 11:02:30 2019

@author: cemsezeroglu
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#veri yükleme
veriler = pd.read_csv("maaslar.csv")
#eğitim seviyesiyle maaşı ilişkendiricez

#Data frame slicing
x_egitim_seviyesi = veriler.iloc[:,1:2]
y_maas = veriler.iloc[:,2:]

#bunlar arasında polinomal ilişki yapabiliriz
#ama yinede lineer kursak ne olur diye bakıcaz

#Linear Regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg=lin_reg.fit(x_egitim_seviyesi.values,y_maas.values)

#görselleştirme için
plt.scatter(x_egitim_seviyesi.values,y_maas.values,color = "red")
plt.plot(x_egitim_seviyesi,lin_reg.predict(x_egitim_seviyesi),color = "blue")
plt.legend(["egitim seviyesi tahmin","egitim vs maas"])
plt.show()


from sklearn.metrics import r2_score
print("Linear Regression R2 değeri : ")
print(r2_score(y_maas,lin_reg.predict(x_egitim_seviyesi)))

#Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures
#PolynomialFeatures herhangi bir sayıyı polinomal olarak ifade etmeye yarıyo
poly_reg = PolynomialFeatures(degree = 2)
#2.derece dedik
x_poly = poly_reg.fit_transform(x_egitim_seviyesi.values)
lin_reg_new = LinearRegression()
lin_reg_new.fit(x_poly,y_maas.values)

#Görselleştirme
plt.scatter(x_egitim_seviyesi.values,y_maas.values,color = "red")
plt.plot(x_egitim_seviyesi.values,lin_reg_new.predict(poly_reg.fit_transform(x_egitim_seviyesi.values)),color = "blue")
plt.legend(["egitim seviyesi tahmin","egitim vs maas"])
plt.show()


from sklearn.preprocessing import PolynomialFeatures
#PolynomialFeatures herhangi bir sayıyı polinomal olarak ifade etmeye yarıyo

poly_reg = PolynomialFeatures(degree = 4)
#4.derece dedik
x_poly = poly_reg.fit_transform(x_egitim_seviyesi.values)
lin_reg_new = LinearRegression()
lin_reg_new.fit(x_poly,y_maas.values)

#Görselleştirme
plt.scatter(x_egitim_seviyesi.values,y_maas.values,color = "red")
plt.plot(x_egitim_seviyesi.values,lin_reg_new.predict(poly_reg.fit_transform(x_egitim_seviyesi.values)),color = "blue")
plt.legend(["egitim seviyesi tahmin","egitim vs maas"])
plt.show()
#4.dereceden olan data ya tam oturdu

#tahminler
print(lin_reg.predict(11))#[[34716.66666667]]
print(lin_reg.predict(6.6))#[[16923.33333333]]

print(lin_reg_new.predict(poly_reg.fit_transform(11))) #[[89041.66666665]]
print(lin_reg_new.predict(poly_reg.fit_transform(6.6))) #[[8146.9948718]]

from sklearn.metrics import r2_score
print("Polynomail regression R2 değeri : ")
print(r2_score(y_maas,lin_reg_new.predict(poly_reg.fit_transform(x_egitim_seviyesi))))
