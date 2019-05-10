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



from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1, metric='minkowski')
knn_pca = KNeighborsClassifier(n_neighbors=1, metric='minkowski')

knn.fit(x_train, y_train)
knn_pca.fit(x_train_pca, y_train)

y_pred_knn = knn.predict(x_test)
y_pred_pca_knn = knn_pca.predict(x_test_pca)

cm4 = confusion_matrix(y_test, y_pred_knn)
cm5 = confusion_matrix(y_test, y_pred_pca_knn)
print("KNN ile PCA'sız")
print(cm4,"\n\n")
print("KNN ile PCA'lı")
print(cm5,"\n\n")



from sklearn.svm import SVC
svc = SVC(kernel='poly')
svc_pca = SVC(kernel='poly')

svc.fit(x_train, y_train)
svc_pca.fit(x_train_pca, y_train)

y_pred_svm = svc.predict(x_test)
y_pred_pca_svm = svc_pca.predict(x_test_pca)

cm6 = confusion_matrix(y_test, y_pred_svm)
cm7 = confusion_matrix(y_test, y_pred_pca_svm)
print("Destek Vektör Makinesi ile PCA'sız")
print(cm6,"\n\n")
print("Destek Vektör Makinesi ile PCA'lı")
print(cm7,"\n\n")



from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb_pca = GaussianNB()

gnb.fit(x_train, y_train)
gnb_pca.fit(x_train_pca, y_train)

y_pred_gnb = gnb.predict(x_test)
y_pred_pca_gnb = gnb_pca.predict(x_test_pca)

cm8 = confusion_matrix(y_test, y_pred_gnb)
cm9 = confusion_matrix(y_test, y_pred_pca_gnb)
print("Bayes ile PCA'sız")
print(cm8,"\n\n")
print("Bayes ile PCA'lı")
print(cm9,"\n\n")



from sklearn.neural_network import MLPClassifier  
mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
mlp_pca = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)

mlp.fit(x_train, y_train)
mlp_pca.fit(x_train_pca, y_train)

y_pred_mlp = mlp.predict(x_test)
y_pred_pca_mlp = mlp_pca.predict(x_test_pca)

cm10 = confusion_matrix(y_test, y_pred_mlp)
cm11 = confusion_matrix(y_test, y_pred_pca_mlp)
print("Yapay Sinir Ağı ile PCA'sız")
print(cm10,"\n\n")
print("Yapay Sinir Ağı ile PCA'lı")
print(cm11,"\n\n")