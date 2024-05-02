"""
## Kelompok 3D
- Alief Cahyo
- Agastya       
- Irfan Eka

## Pengenalan

K-Nearest Neighbors (KNN) adalah salah satu algoritma yang sederhana dan intuitif dalam klasifikasi dan regresi. Algoritma ini bekerja dengan cara mencari K titik data terdekat dalam ruang fitur untuk melakukan prediksi atau klasifikasi terhadap data yang baru. KNN sering digunakan dalam berbagai bidang seperti pengenalan pola, analisis teks, dan bioinformatika.

## Langkah-langkah dalam Menggunakan KNN

1. Penentuan Parameter K: Pilih nilai K yang sesuai untuk jumlah tetangga terdekat yang akan digunakan dalam prediksi.
2. Pemilihan Metrik Jarak: Tentukan metrik jarak yang akan digunakan untuk mengukur kedekatan antar data.
3. Pembagian Data: Pisahkan dataset menjadi data latih (train data) dan data uji (test data).
4. Pembuatan Model: Bangun model KNN menggunakan data latih.
5. Klasifikasi atau Prediksi: Gunakan model yang telah dibuat untuk melakukan klasifikasi atau prediksi pada data uji.
6. Evaluasi Model: Evaluasi kinerja model menggunakan data uji.

## Implementasi dengan Python
"""

# Import library
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load dataset iris
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Pisahkan data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Bangun model KNN
model = KNeighborsClassifier(n_neighbors=3)  # Misalnya, pilih nilai K=3
model.fit(X_train, y_train)

# Lakukan prediksi
y_pred = model.predict(X_test)

# Evaluasi model
accuracy = accuracy_score(y_test, y_pred)
print("Akurasi:", accuracy)
