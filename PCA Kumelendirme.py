import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

csv_dosya = None
veriler = pd.read_csv(csv_dosya)
x = veriler.iloc[:, 1:8].values
y = veriler.iloc[:, 0].values


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


from sklearn.decomposition import PCA
pca = PCA(n_components = 2)

x_train_pca = pca.fit_transform(x_train)
x_test_pca = pca.transform(x_test)


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier_pca = LogisticRegression(random_state=0)

classifier.fit(x_train,y_train)
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

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=0)
kmeans_pca = KMeans(n_clusters=3, random_state=0)

kmeans.fit(x_train)
kmeans_pca.fit(x_train_pca)

y_pred_kmn =  kmeans.predict(x_test)
y_pred_pca_kmn = kmeans_pca.predict(x_test_pca)

cm8 = confusion_matrix(y_test, y_pred_kmn)
cm9 = confusion_matrix(y_test, y_pred_pca_kmn)
print("KMeans ile pca'sız")
print(cm8,"\n\n")
print("KMeans ile pca'lı")
print(cm9,"\n\n")

