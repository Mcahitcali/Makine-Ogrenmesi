# -*- coding: utf-8 -*-

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#1. kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# veri kümesi
#veriler = pd.read_csv('D:\iris.csv')
#x = veriler.iloc[:, 0:4].values
#y = veriler.iloc[:, 4].values
#y = veriler.loc[:,"Cinsiyet"]

#veriler = pd.read_csv('D:\denizkulagid.csv')
#x = veriler.iloc[:, 1:].values
#y = veriler.iloc[:, 0].values
csv_dosya = None
veriler = pd.read_csv(csv_dosya)
x = veriler.iloc[:, 0:13].values
y = veriler.iloc[:, 13].values

print(veriler.head())

print(veriler.describe())

print(veriler.corr())


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test) #fit_transform


from sklearn.decomposition import PCA
pca = PCA(n_components = 2)

x_train_pca = pca.fit_transform(x_train)
x_test_pca = pca.transform(x_test)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(x_train,y_train)



classifier_pca = LogisticRegression(random_state=0)
classifier_pca.fit(x_train_pca,y_train)


y_pred = classifier.predict(x_test)
y_pred_pca = classifier_pca.predict(x_test_pca)

from sklearn.metrics import confusion_matrix

print("\nPCA Olmadan Çıkan Sonuç")
cm = confusion_matrix(y_test,y_pred)
print(cm,"\n\n")


print("PCA Sonrası Çıkan Sonuç")
cm2 = confusion_matrix(y_test,y_pred_pca)
print(cm2,"\n\n")

##PCA sonrası / PCA öncesi
#print("PCA Öncesi / PCA Sonrası")
#cm3 = confusion_matrix(y_pred,y_pred_pca)
#print(cm3,"\n\n")

from matplotlib.colors import ListedColormap

def plot_decision_regions(X, y, classifier, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
            alpha=0.8, c=cmap(idx),
            marker=markers[idx], label=cl)


plot_decision_regions(x_train_pca, y_train, classifier=classifier_pca)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(loc='lower left')
plt.show()