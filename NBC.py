#Завантажуємо датасет
from sklearn.datasets import load_wine
import pandas as pd
import numpy as np
from scipy.stats import norm


data = load_wine()

X,y,column_names= data['data'], data['target'], data['feature_names']
X=pd.DataFrame(X,columns =column_names)

#Розділення даних
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X,y,random_state=44)

#Статистика для тренувальної множини
means = X_train.groupby(y_train).apply(np.mean)
stds=X_train.groupby(y_train).apply(np.std)

#Докласові ймовірності
probs = X_train.groupby(y_train).apply(lambda x: len(x))/X_train.shape[0]

y_pred = []
#Для кожного елемента перевірочної мн
for elem in range(X_val.shape[0]):
    p={}
    #для будь-якого можливого класу
    for cl in np.unique(y_train):
        #беремо ймовірність для цього класу
        p[cl] = probs.iloc[cl]
        #для кожного стовпця
        for index, param in enumerate(X_val.iloc[elem]):
            # множимо на значення ймовірністі заданого сповпця
            # розподілу тренувального стовпця заданого класу
            p[cl] *= norm.pdf(param, means.iloc[cl,index],stds.iloc[cl,index])
    y_pred.append(pd.Series(p).values.argmax())

#Результуючий класифікатор
from sklearn.metrics import accuracy_score
print(accuracy_score(y_val, y_pred))
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
gnb= GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
print(gnb.predict([[12,2,1,20,95,280,2,0.3,519,3.1,0.36,1.64,2.8]]))
y_pred = gnb.fit(X_train,y_train).predict(X_test)