#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 13:52:58 2019

@author: cemsezeroglu
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

veriler = pd.read_csv("maaslar.csv")

x = veriler.iloc[:,1:2]
y = veriler.iloc[:,2:]



from sklearn.tree import DecisionTreeRegressor

r_dt = DecisionTreeRegressor(random_state = 0)

r_dt.fit(x,y)

plt.scatter(x,y,color="red")
plt.plot(x,r_dt.predict(x),color = "blue")
plt.show()

print("decision tree tahmin")

print(r_dt.predict(50))
print(r_dt.predict(6.6))

from sklearn.metrics import r2_score
print("decision tree R2 değeri : ")
print(r2_score(y,r_dt.predict(x)))

#decision tree R2 değeri : 
#1.0
#her değer için aynı aralığı döndürüyo mükemmel sonuç veriyo 
#ama farklı sonuç verdiğimizde aynı sonuçları döndürüyo
#r2 burda kullanımı tehlikeli yanıltıyor
