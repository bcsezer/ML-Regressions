#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 04:18:20 2018

@author: sadievrenseker
"""

#1. kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2. Veri Onisleme

#2.1. Veri Yukleme
veriler = pd.read_csv('veriler.csv')
#pd.read_csv("veriler.csv")


#veri on isleme



#eksik veriler

#encoder:  Kategorik -> Numeric
ulke = veriler.iloc[:,0:1].values
print(ulke)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
ulke[:,0] = le.fit_transform(ulke[:,0])
print(ulke)

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features='all')
ulke=ohe.fit_transform(ulke).toarray()
print(ulke)


c= veriler.iloc[:,-1:].values
print(c)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
c[:,0] = le.fit_transform(c[:,0])
print(c)

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features='all')
c=ohe.fit_transform(c).toarray()
print(c)

Yas = veriler.iloc[:,1:4].values  

#numpy dizileri dataframe donusumu
sonuc = pd.DataFrame(data = ulke, index = range(22), columns=['fr','tr','us'] )
print(sonuc)

sonuc2 =pd.DataFrame(data = Yas, index = range(22), columns = ['boy','kilo','yas'])
print(sonuc2)

cinsiyet = veriler.iloc[:,-1].values
print(cinsiyet)

sonuc3 = pd.DataFrame(data = c[:,:1], index=range(22), columns=['cinsiyet'])
print(sonuc3)

#dataframe birlestirme islemi
s=pd.concat([sonuc,sonuc2],axis=1)
print(s)

s2= pd.concat([s,sonuc3],axis=1)
print(s2)

#verilerin egitim ve test icin bolunmesi
from sklearn.cross_validation import train_test_split
x_train, x_test,y_train,y_test = train_test_split(s,sonuc3,test_size=0.33, random_state=0)

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)


from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)

boy = s2.iloc[:,3:4].values

sol = s2.iloc[:,:3]
sag = s2.iloc[:,4:]

veri = pd.concat([sol,sag],axis = 1)



#boy y bağımlı değişken oldu yalnız bıraktık
#bağımsız değişkenleri kullanarak veri kümesini böldük

x_train, x_test,y_train,y_test = train_test_split(veri,boy,test_size=0.33, random_state=0)


r2 = LinearRegression()
r2.fit(x_train,y_train)

y_pred = r2.predict(x_test)

#p-value ve Backward elimination 

import statsmodels.formula.api as sm

#burada ß değerini dataframe e eklemeye çalışıyoruz en başına ya da en sonuna 
# ones dediğimiz bir tane birlerden oluşan dizi oluşturur.


X = np.append(arr =np.ones((22,1)).astype(int),values = veri,axis=1)

#veri data frameminde ki her bir kolonu ifade edicek bir liste oluşturucaz
#tamamnı ilk başta alıp p-value hesaplıycaz o yüzden aldık.
X_l = veri.iloc[:,[0,1,2,3,5]].values

r = sm.OLS(endog = boy , exog =X_l ).fit()#boy ile sonuçta bulmak istediğimiz boy
#bağımlı değişkenimiz ve bağımsız değişkenleri içeren dizi arasındaki teker teker bağlantıyı bulma
#1 in 2 nin ... boy üzerinde ki etkisini ölçmek.

print(r.summary()) # burada bir rapor vericek bize 
#R-squared:,Adj. R-squared: ÖNEMLİ DEĞERLER
# AMA BİZ P>|t| DEĞERİNE BAKICAZ burda x5 in yani 4. elamanın değeri yüksek.
#daha sonra yukarıda oluşturduğumuz listeden X_l = veri.iloc[:,[0,1,2,3,4,5]].values normalde böyleydi
#4. elemanı elemek

    
    

