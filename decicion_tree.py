"""
## Kemlompok 3D
- Alief Cahyo
- Agastya       
- Irfan Eka

## Pengenalan

Decision Tree adalah salah satu algoritma yang umum digunakan dalam data mining dan machine learning untuk melakukan klasifikasi dan regresi. Algoritma ini membagi dataset menjadi subset yang lebih kecil berdasarkan fitur-fitur yang ada, dengan tujuan untuk membuat keputusan yang optimal di setiap langkahnya.

## Langkah-langkah dalam Menggunakan Decision Tree

1. Pemilihan Fitur: Identifikasi fitur-fitur yang akan digunakan untuk membuat model Decision Tree.
2. Pembagian Data: Pisahkan dataset menjadi data latih (train data) dan data uji (test data).
3. Pembuatan Model: Bangun model Decision Tree menggunakan data latih.
4. Evaluasi Model: Evaluasi kinerja model menggunakan data uji.

## Implementasi dengan Python
"""

import pandas as pd
from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load Iris Dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create Decision Tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)

# Make Predictions
y_pred = clf.predict(X_test)

# Evaluate Model
accuracy = accuracy_score(y_test, y_pred)
print('Akurasi Prediksi: {:.2f}'.format(accuracy))


