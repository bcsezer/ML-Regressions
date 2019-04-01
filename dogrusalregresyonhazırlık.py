#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 04:18:20 2018

@author: cemsezeroglu
"""

#1. kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2. Veri Onisleme

#2.1. Veri Yukleme
veriler = pd.read_csv('satislar.csv')
#pd.read_csv("veriler.csv")


#veri on isleme
aylar = veriler[['Aylar']]
#test
print(aylar)

satislar = veriler[['Satislar']]
print(satislar)



#verilerin egitim ve test icin bolunmesi
from sklearn.cross_validation import train_test_split
x_train, x_test,y_train,y_test = train_test_split(aylar,satislar,test_size=0.33, random_state=0)
#satışlar aylara bağımlı -------------------------------^)




#verilerin olceklenmesi
'''
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

Y_train = sc.fit_transform(y_train)
Y_test = sc.fit_transform(y_test)
#burada verileri standardlaştırdık.
'''


#model insası linear regression
from sklearn.linear_model import LinearRegression
#test ve train verilerimiz hazır şimdi lineer regresyon modelini inşa ediyoruz.

lr = LinearRegression()

lr.fit(x_train,y_train)

tahmin = lr.predict(x_test)

#görselleştirme 
x_train = x_train.sort_index() #verileri random bir biçimde dağıttığımız için grafiği saçma olmasın diye tekarardan sort ediyoruz.
y_train = y_train.sort_index() 

plt.plot(x_train,y_train) #bu verilen orjinal rastgele yüzde 67 bölünmüş hali

plt.plot(x_test,lr.predict(x_test))#teker teker her bir test değerine karşılık gelen tahmin değeri.

plt.title("aylara göre satış")
plt.xlabel("Aylar")
plt.ylabel("Satışlar")







    
    

