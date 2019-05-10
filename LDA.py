
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


veriler = pd.read_csv('D:\denizkulagi.csv')
x = veriler.iloc[:, 1:].values
y = veriler.iloc[:, 0].values


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components = 2)

x_train_lda = lda.fit_transform(x_train,y_train)
x_test_lda = lda.transform(x_test)


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(x_train,y_train)


classifier_lda = LogisticRegression(random_state=0)
classifier_lda.fit(x_train_lda,y_train)


y_pred = classifier.predict(x_test)
y_pred_lda = classifier_lda.predict(x_test_lda)

from sklearn.metrics import confusion_matrix

print("\nLDA Olmadan Çıkan Sonuç")
cm = confusion_matrix(y_test,y_pred)
print(cm,"\n\n")

print("LDA Sonrası Çıkan Sonuç")
cm2 = confusion_matrix(y_test,y_pred_lda)
print(cm2,"\n\n")

print("LDA Öncesi / LDA Sonrası")
cm3 = confusion_matrix(y_pred,y_pred_lda)
print(cm3,"\n\n")