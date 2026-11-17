📑 Daftar Isi
Tentang Proyek
Tujuan Pembelajaran
Isi Repositori
Dasar Teori Singkat
Dataset
Alur Kerja Setiap Notebook
Hyperparameter yang Digunakan
Metrik Evaluasi
Teknologi yang Digunakan
Instalasi dan Cara Menjalankan
Contoh Penggunaan Model
Hasil
Struktur Folder
Catatan dan Keterbatasan
Pengembangan Selanjutnya
Troubleshooting
Kontribusi
Lisensi
Penulis
📖 Tentang Proyek

Repositori ini berisi kumpulan notebook latihan Support Vector Machine (SVM) menggunakan Scikit-Learn. SVM adalah algoritma supervised learning yang dapat digunakan untuk dua jenis tugas:

Tugas	Algoritma	Studi Kasus
Regresi	Support Vector Regression (SVR)	Memprediksi gaji berdasarkan lama pengalaman kerja
Klasifikasi	Support Vector Classifier (SVC)	Memprediksi apakah pasien menderita diabetes

Seluruh kode ditulis dalam Jupyter Notebook sehingga mudah dibaca, dijalankan sel demi sel, dan dimodifikasi untuk bereksperimen.

🎯 Tujuan Pembelajaran

Setelah mempelajari repositori ini, Anda diharapkan mampu:

Memahami perbedaan SVR (regresi) dan SVC (klasifikasi).
Membaca dan mengeksplorasi dataset dengan pandas.
Melakukan standarisasi data dengan StandardScaler dan memahami mengapa SVM membutuhkannya.
Membagi data latih dan data uji dengan train_test_split.
Melatih model SVM dan mengatur hyperparameter (C, gamma, epsilon, kernel).
Mengevaluasi model regresi (MAE, RMSE, R²) dan klasifikasi (akurasi).
Memvisualisasikan hasil prediksi dengan Matplotlib.
Menghindari data leakage pada proses preprocessing.
📂 Isi Repositori
Notebook	Tugas	Model	Dataset	Level
Latihan_SKLearn_SVR.ipynb	Regresi (dasar)	SVR kernel RBF	Salary_Data.csv	Pemula
SVMRegresi__1_.ipynb	Regresi (lengkap + evaluasi)	SVR kernel RBF	Salary_Data.csv	Menengah
SVM_Klasifikasi_DIabetes.ipynb	Klasifikasi	SVC	diabetes.csv	Pemula

Urutan belajar yang disarankan: Latihan_SKLearn_SVR → SVMRegresi__1_ → SVM_Klasifikasi_DIabetes

🧠 Dasar Teori Singkat
Support Vector Machine (SVM)

SVM mencari hyperplane (batas keputusan) terbaik yang memisahkan data dengan margin selebar mungkin. Titik data yang berada paling dekat dengan hyperplane disebut support vectors, dan titik-titik inilah yang menentukan posisi hyperplane.

Support Vector Classifier (SVC)

Digunakan untuk klasifikasi. Model mencari batas yang memisahkan kelas-kelas data dengan margin maksimum, dengan toleransi kesalahan yang diatur oleh parameter C.

Support Vector Regression (SVR)

Digunakan untuk regresi. Alih-alih memisahkan kelas, SVR mencari fungsi yang menyesuaikan data dalam tabung toleransi (ε-tube). Titik yang berada di dalam tabung tidak dihitung sebagai error; hanya titik di luar tabung yang memengaruhi model.

Kernel Trick

Kernel memungkinkan SVM menangani data yang tidak terpisah secara linear dengan memetakannya ke ruang berdimensi lebih tinggi. Kernel RBF (Radial Basis Function) dipakai pada proyek ini karena mampu menangkap pola non-linear.

Mengapa Perlu Standarisasi?

SVM berbasis jarak antar titik data, sehingga fitur dengan skala besar bisa mendominasi fitur berskala kecil. StandardScaler mengubah data sehingga memiliki rata-rata 0 dan standar deviasi 1.

📊 Dataset

⚠️ File dataset harus diletakkan di folder yang sama dengan notebook.

1. Salary_Data.csv

Data hubungan antara lama pengalaman kerja dan gaji.

Kolom	Tipe	Keterangan
YearsExperience	Numerik (float)	Lama pengalaman kerja (tahun), digunakan sebagai fitur
Salary	Numerik	Gaji, digunakan sebagai target
2. diabetes.csv

Dataset diabetes (umumnya dikenal sebagai Pima Indians Diabetes Dataset) dengan 8 fitur medis dan 1 label.

Kolom	Keterangan
Pregnancies	Jumlah kehamilan
Glucose	Kadar glukosa plasma
BloodPressure	Tekanan darah diastolik
SkinThickness	Ketebalan lipatan kulit trisep
Insulin	Kadar insulin serum
BMI	Indeks massa tubuh
DiabetesPedigreeFunction	Skor riwayat diabetes dalam keluarga
Age	Usia (tahun)
Outcome	Label: 0 = tidak diabetes, 1 = diabetes
🔄 Alur Kerja Setiap Notebook
1️⃣ Latihan_SKLearn_SVR.ipynb — SVR Dasar
Baca data → Pisahkan X & y → Ubah X ke array 2D → Latih SVR → Visualisasi
Membaca Salary_Data.csv dengan pd.read_csv().
Menampilkan informasi dataset dengan data.info() dan data.head().
Memisahkan fitur YearsExperience dan target Salary.
Mengubah fitur menjadi array 2D (X[:, np.newaxis]) sesuai format yang dibutuhkan Scikit-Learn.
Melatih model SVR(C=1000, gamma=0.05, kernel='rbf').
Memvisualisasikan data asli (titik biru) dan kurva prediksi SVR (garis merah).
2️⃣ SVMRegresi__1_.ipynb — SVR dengan Evaluasi Lengkap
Baca data → Split 80:20 → Scaling (fit di train saja) → Latih SVR
→ Prediksi → Inverse transform → Evaluasi (MAE, RMSE, R²) → Visualisasi → Prediksi baru
Memuat data dan memilih YearsExperience sebagai fitur, Salary sebagai target.
Membagi data: 80% latih, 20% uji (random_state=42).
Standarisasi fitur (scaler_X) dan target (scaler_y). Scaler hanya di-fit pada data latih untuk mencegah data leakage.
Melatih SVR(kernel="rbf", C=100, epsilon=0.1, gamma="scale").
Mengembalikan hasil prediksi ke satuan gaji asli dengan inverse_transform.
Menghitung metrik MAE, RMSE, dan R².
Menampilkan tabel perbandingan gaji aktual vs prediksi.
Memvisualisasikan data latih, data uji, dan kurva SVR.
Memprediksi gaji untuk pengalaman kerja baru (contoh: 5 tahun).
3️⃣ SVM_Klasifikasi_DIabetes.ipynb — Klasifikasi Diabetes
Baca data → Pisahkan X & y → Standarisasi → Split 67:33 → Latih SVC → Hitung akurasi
Membaca diabetes.csv, lalu menampilkan head() dan info().
Memisahkan 8 kolom pertama sebagai fitur (X) dan Outcome sebagai label (y).
Standarisasi fitur dengan StandardScaler.
Membagi data: 67% latih, 33% uji (random_state=42).
Melatih SVC() dengan parameter bawaan (kernel RBF).
Menampilkan skor akurasi dengan clf.score(X_test, y_test).
⚙️ Hyperparameter yang Digunakan
Notebook	Model	Parameter
Latihan_SKLearn_SVR	SVR	kernel='rbf', C=1000, gamma=0.05
SVMRegresi__1_	SVR	kernel='rbf', C=100, epsilon=0.1, gamma='scale'
SVM_Klasifikasi_DIabetes	SVC	Parameter default (kernel='rbf', C=1.0, gamma='scale')

Penjelasan parameter:

Parameter	Fungsi	Efek
kernel	Jenis fungsi kernel	linear, poly, rbf, sigmoid
C	Penalti terhadap error	Nilai besar → model lebih ketat terhadap data latih (risiko overfitting); nilai kecil → model lebih halus (risiko underfitting)
gamma	Jangkauan pengaruh satu titik data pada kernel RBF	Nilai besar → kurva lebih lentur dan rumit; nilai kecil → kurva lebih halus
epsilon	Lebar tabung toleransi (khusus SVR)	Nilai besar → toleransi error lebih lebar, model lebih sederhana
📏 Metrik Evaluasi
Regresi
Metrik	Rumus	Interpretasi
MAE	rata-rata |y − ŷ|	Rata-rata selisih absolut prediksi dan nilai aktual (dalam satuan gaji)
RMSE	√(rata-rata (y − ŷ)²)	Mirip MAE namun lebih sensitif terhadap error besar
R²	1 − SSres/SStot	Proporsi variansi yang dijelaskan model (semakin mendekati 1 semakin baik)
Klasifikasi
Metrik	Interpretasi
Akurasi	Persentase prediksi yang benar dari seluruh data uji
🛠️ Teknologi yang Digunakan
Library	Kegunaan
Python 3.8+	Bahasa pemrograman
NumPy	Operasi array dan numerik
pandas	Manipulasi dan analisis data
Matplotlib	Visualisasi data
Scikit-Learn	Model SVM, preprocessing, dan metrik evaluasi
Jupyter	Lingkungan menjalankan notebook
🚀 Instalasi dan Cara Menjalankan
Prasyarat
Python 3.8 atau lebih baru
pip (package manager Python)
Git (opsional, untuk clone repositori)
Langkah-langkah

1. Clone repositori

bash
git clone https://github.com/muhammadharistkhairullah-gif/<nama-repositori>.git
cd <nama-repositori>

2. (Opsional) Buat virtual environment

bash
# Linux / macOS
python -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate

3. Install dependensi

bash
pip install -r requirements.txt

atau secara manual:

bash
pip install numpy pandas matplotlib scikit-learn jupyter

4. Siapkan dataset

Letakkan Salary_Data.csv dan diabetes.csv di folder yang sama dengan notebook.

5. Jalankan Jupyter

bash
jupyter notebook

Buka notebook yang diinginkan, lalu pilih Kernel → Restart & Run All atau jalankan sel satu per satu dengan Shift + Enter.

☁️ Alternatif: Google Colab
Buka Google Colab dan unggah notebook (File → Upload notebook).
Unggah file dataset ke sesi Colab (ikon folder di sidebar kiri → Upload).
Jalankan seluruh sel.
💡 Contoh Penggunaan Model
Memprediksi gaji (SVR)
python
import numpy as np

# Prediksi gaji untuk pengalaman kerja 5 tahun
pengalaman_baru = np.array([[5.0]])
pengalaman_baru_scaled = scaler_X.transform(pengalaman_baru)

prediksi_scaled = model_svr.predict(pengalaman_baru_scaled).reshape(-1, 1)
prediksi_gaji = scaler_y.inverse_transform(prediksi_scaled)[0, 0]

print(f"Perkiraan gaji: {prediksi_gaji:,.2f}")
Memprediksi diabetes (SVC)

Contoh berikut menunjukkan cara memakai model yang sudah dilatih untuk satu pasien baru. Urutan nilai harus sama dengan urutan kolom fitur pada diabetes.csv.

python
import numpy as np

# Pregnancies, Glucose, BloodPressure, SkinThickness,
# Insulin, BMI, DiabetesPedigreeFunction, Age
pasien_baru = np.array([[2, 120, 70, 25, 80, 28.5, 0.45, 33]])

pasien_scaled = scaler.transform(pasien_baru)   # scaler yang sama dengan saat training
hasil = clf.predict(pasien_scaled)[0]

print("Diabetes" if hasil == 1 else "Tidak diabetes")

Nilai pada contoh di atas hanya ilustrasi, bukan data pasien sebenarnya.

📈 Hasil

Jalankan notebook untuk mendapatkan nilai pada mesin Anda, lalu isi tabel berikut.

Notebook	Metrik	Hasil
SVMRegresi__1_	MAE	isi setelah menjalankan notebook
SVMRegresi__1_	RMSE	isi setelah menjalankan notebook
SVMRegresi__1_	R²	isi setelah menjalankan notebook
SVM_Klasifikasi_DIabetes	Akurasi	isi setelah menjalankan notebook

Anda juga dapat menambahkan tangkapan layar grafik ke repositori dan menampilkannya di sini:

markdown
![Kurva SVR](images/kurva_svr.png)
🗂️ Struktur Folder
.
├── Latihan_SKLearn_SVR.ipynb          # SVR dasar
├── SVMRegresi__1_.ipynb               # SVR dengan evaluasi lengkap
├── SVM_Klasifikasi_DIabetes.ipynb     # Klasifikasi diabetes dengan SVC
├── Salary_Data.csv                    # Dataset gaji
├── diabetes.csv                       # Dataset diabetes
├── requirements.txt                   # Daftar dependensi
└── README.md                          # Dokumentasi proyek
⚠️ Catatan dan Keterbatasan
Data leakage pada notebook diabetes. Scaler di-fit pada seluruh data sebelum dilakukan pembagian data latih dan uji. Praktik yang lebih tepat adalah melakukan split terlebih dahulu, lalu fit scaler hanya pada data latih (seperti pada SVMRegresi__1_.ipynb), atau menggunakan Pipeline.
Notebook SVR dasar melatih model pada seluruh data tanpa pembagian data uji, sehingga tidak ada evaluasi performa pada data baru. Gunakan SVMRegresi__1_.ipynb untuk evaluasi yang benar.
Ukuran dataset kecil. Dataset gaji berukuran kecil sehingga hasil evaluasi bisa sangat bergantung pada pembagian data.
Akurasi saja belum cukup. Pada data medis, metrik seperti precision, recall, dan F1-score sangat penting, terutama bila kelas tidak seimbang.
Tujuan edukasi. Model pada repositori ini hanya untuk pembelajaran dan tidak boleh digunakan sebagai alat diagnosis medis.
Hyperparameter belum dioptimalkan secara sistematis.
🔭 Pengembangan Selanjutnya
[ ] Menambahkan Pipeline (scaler + model) untuk mencegah data leakage
[ ] Tuning hyperparameter dengan GridSearchCV / RandomizedSearchCV
[ ] Membandingkan kernel linear, poly, rbf, dan sigmoid
[ ] Menambahkan confusion matrix, classification report, dan kurva ROC pada klasifikasi diabetes
[ ] Melakukan cross-validation untuk evaluasi yang lebih stabil
[ ] Menangani nilai nol yang tidak wajar pada kolom medis (misalnya Glucose, BloodPressure, BMI)
[ ] Membandingkan SVM dengan model lain (Logistic Regression, Random Forest, KNN)
[ ] Menyimpan model dengan joblib dan membuat demo aplikasi sederhana (Streamlit / Gradio)
🩺 Troubleshooting
Masalah	Solusi
FileNotFoundError: Salary_Data.csv / diabetes.csv	Pastikan file CSV berada di folder yang sama dengan notebook, atau ubah path pada pd.read_csv().
ModuleNotFoundError: No module named 'sklearn'	Jalankan pip install scikit-learn.
ValueError: Expected 2D array, got 1D array	Ubah fitur menjadi 2D, misalnya X = data[['YearsExperience']] atau X[:, np.newaxis].
Grafik tidak muncul	Tambahkan %matplotlib inline di awal notebook.
Hasil berbeda dengan yang di README	Pastikan random_state dan versi library sama.
🤝 Kontribusi

Kontribusi sangat diterima! Cara berkontribusi:

Fork repositori ini
Buat branch baru: git checkout -b fitur/nama-fitur
Commit perubahan: git commit -m "Menambahkan fitur ..."
Push ke branch: git push origin fitur/nama-fitur
Buka Pull Request
📄 Lisensi

Proyek ini dirilis di bawah lisensi MIT. Lihat berkas LICENSE untuk detail lebih lanjut.

👤 Penulis

Muhammad Harist Khairullah

GitHub: @muhammadharistkhairullah-gif
Email: muhammadharistkhairullah@gmail.com
