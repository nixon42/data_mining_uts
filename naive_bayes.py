"""
## Kelompok 3D
- ALief Cahyo
- Agastya
- Irfan Eka

## Pengenalan

Naive Bayes adalah salah satu algoritma klasifikasi yang berbasis probabilitas. Algoritma ini sangat berguna dalam melakukan klasifikasi data berlabel, seperti klasifikasi teks dalam analisis sentimen, klasifikasi dokumen, dan lain sebagainya. Meskipun sederhana, Naive Bayes sering kali memberikan hasil yang baik dalam berbagai kasus.

## Langkah-langkah dalam Menggunakan Naive Bayes

1. Pemilihan Fitur: Identifikasi fitur-fitur yang relevan untuk klasifikasi.
2. Pembagian Data: Pisahkan dataset menjadi data latih (train data) dan data uji (test data).
3. Pembuatan Model: Pelajari distribusi probabilitas fitur-fitur pada setiap kelas.
4. Klasifikasi: Gunakann model yang telah dibuat untuk melakukan klasifikasi pada data uji.
5. Evaluasi Model: Evaluasi kinerja model menggunakan data uji.

## Implementasi dengan Python
"""
# Import libraries
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

# Load Iris Dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target_names[iris.target])

# Split data to train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create classifier: Gaussian Naive Bayes
clf = GaussianNB()

# Train classifier
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Print accuracy
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

