# Proyek UTS: Prediksi dengan Regresi dan Evaluasi MAE, MSE, R²

Proyek ini membandingkan tiga model regresi pada dataset **Rice (Cammeo & Osmancik)**, dengan dataset **Wine Quality** disiapkan sebagai dataset kedua. Kinerja model diukur dengan tiga metrik: **MAE**, **MSE**, dan **R²**.

> **Jawaban singkat untuk pertanyaan dosen**
>
> | Metrik | Nilai terbaik | Arah yang baik |
> |---|---|---|
> | **MAE** | **0** | Semakin kecil semakin baik. **Bukan** mendekati 1 |
> | **MSE** | **0** | Semakin kecil semakin baik |
> | **R²** | **1** | Semakin mendekati 1 semakin baik |

---

## Daftar Isi

1. [Gambaran Proyek](#1-gambaran-proyek)
2. [Landasan Teori](#2-landasan-teori)
3. [Alur Pengerjaan](#3-alur-pengerjaan)
4. [Hasil Evaluasi](#4-hasil-evaluasi)
5. [Cara Menjalankan](#5-cara-menjalankan)
6. [Catatan Metodologi](#6-catatan-metodologi)
7. [Referensi](#7-referensi)

---

## 1. Gambaran Proyek

| Item | Keterangan |
|---|---|
| Berkas utama | `projectuts.ipynb` |
| Dataset 1 | `Rice_Cammeo_Osmancik.csv` (fitur morfologi butir beras, target `Class`) |
| Dataset 2 | `winequality.csv` (pemisah `;`, target `quality`) |
| Model | Least Angle Regression (LARS), Linear Regression, Gradient Boosting Regressor |
| Metrik | MAE, MSE, R² (`sklearn.metrics`) |
| Pembagian data | 80% latih, 20% uji, `random_state=1` |

Fitur pada dataset beras: `Area`, `Perimeter`, `Major_Axis_Length`, `Minor_Axis_Length`, `Eccentricity`, `Convex_Area`, `Extent`.

---

## 2. Landasan Teori

Pada regresi, model memprediksi nilai kontinu. Karena itu kualitasnya diukur dari **seberapa jauh prediksi menyimpang dari nilai aktual**, bukan dari benar atau salahnya kategori seperti pada klasifikasi. Untuk n data uji dengan nilai aktual yᵢ, prediksi ŷᵢ, dan rata-rata aktual ȳ, ketiga metrik didefinisikan sebagai berikut.

### 2.1 Mean Absolute Error (MAE)

$$MAE = \frac{1}{n}\sum_{i=1}^{n}\left|y_i-\hat{y}_i\right|$$

- MAE adalah rata-rata selisih **mutlak** antara nilai aktual dan prediksi.
- Rentang nilai: **0 sampai +∞**. Nilai 0 berarti prediksi sempurna.
- **MAE yang baik adalah yang mendekati 0, bukan mendekati 1.** Nilai 1 tidak punya makna khusus sebagai "terbaik".
- Satuannya **sama dengan satuan target**. MAE 0,5 pada target skala 3 sampai 9 (misalnya kualitas wine) punya arti yang berbeda dari MAE 0,5 pada target dalam jutaan rupiah.
- Perlakuannya linear terhadap error, jadi **tidak terlalu sensitif terhadap outlier** dibanding MSE.

### 2.2 Mean Squared Error (MSE)

$$MSE = \frac{1}{n}\sum_{i=1}^{n}\left(y_i-\hat{y}_i\right)^2$$

- MSE adalah rata-rata **kuadrat** selisih antara aktual dan prediksi.
- Rentang nilai: **0 sampai +∞**. Nilai 0 berarti prediksi sempurna.
- **Semakin kecil semakin baik.**
- Karena dikuadratkan, error besar dihukum jauh lebih berat, sehingga MSE **sensitif terhadap outlier**.
- Satuannya adalah kuadrat satuan target. Untuk mengembalikan ke satuan asli dipakai akar MSE (RMSE).

### 2.3 Koefisien Determinasi (R²)

$$R^2 = 1-\frac{SS_{res}}{SS_{tot}} = 1-\frac{\sum_i (y_i-\hat{y}_i)^2}{\sum_i (y_i-\bar{y})^2}$$

- R² menyatakan **proporsi variasi target yang dapat dijelaskan oleh model**.
- **R² yang baik adalah yang mendekati 1.** R² = 1 berarti model cocok sempurna (setara MSE = 0).
- R² = 0 berarti model **tidak lebih baik daripada sekadar memprediksi rata-rata** target.
- R² bisa **negatif** jika MSE model lebih besar daripada MSE prediksi rata-rata, artinya model lebih buruk daripada baseline tersebut. Hal ini bisa terjadi pada data uji.
- R² tidak bergantung pada skala data, sehingga lebih mudah dibandingkan antar dataset dibanding MAE dan MSE.

### 2.4 Ringkasan Interpretasi

| Metrik | Jenis | Rentang | Terbaik | Peka outlier | Bergantung skala target |
|---|---|---|---|---|---|
| MAE | Error (semakin kecil semakin baik) | [0, ∞) | 0 | Rendah | Ya |
| MSE | Error (semakin kecil semakin baik) | [0, ∞) | 0 | Tinggi | Ya (kuadrat) |
| R² | Goodness of fit (semakin besar semakin baik) | (−∞, 1] | 1 | Sedang | Tidak |

### 2.5 Catatan Penting: Kapan "mendekati 1" bisa menyesatkan

Pada notebook ini, kolom `Class` pada dataset beras (Cammeo atau Osmancik) diubah dengan `LabelEncoder` menjadi **0 dan 1**. Akibatnya selisih mutlak antara prediksi dan aktual umumnya berada di antara 0 dan 1, sehingga **MAE ikut berada pada rentang 0 sampai 1**. Meski begitu, **interpretasinya tetap sama**: MAE 0,05 jauh lebih baik daripada MAE 0,5. Nilai MAE yang mendekati 1 pada target biner justru berarti model hampir selalu salah.

Perlu diingat pula bahwa R² yang tinggi tidak otomatis berarti model bagus pada data baru. R² hanya mengukur kecocokan, sehingga risiko overfitting tetap harus diperiksa dengan data uji atau validasi silang.

### 2.6 Mengapa memakai ketiga metrik sekaligus

Tidak ada satu metrik yang selalu terbaik. MAE memberi gambaran besar kesalahan yang mudah dibaca, MSE menonjolkan kesalahan besar, dan R² menunjukkan seberapa banyak variasi data yang berhasil dijelaskan. Chicco dkk. (2021) menyimpulkan bahwa R² lebih informatif dibanding SMAPE, MAE, MAPE, MSE, dan RMSE karena nilai R² sendiri sudah menunjukkan baik buruknya model tanpa perlu mengetahui rentang nilai data. Kendati begitu, pelaporan ketiganya bersama-sama memberi gambaran yang lebih lengkap.

---

## 3. Alur Pengerjaan

1. **Data loading.** Membaca berkas ARFF/CSV beras dengan melewati 16 baris komentar dan mendefinisikan nama kolom secara manual.
2. **Data cleaning.**
   - Memeriksa informasi, statistik deskriptif, dan nilai hilang.
   - Imputasi median untuk kolom numerik dan modus untuk kolom kategorikal.
   - Deteksi outlier dengan boxplot dan aturan IQR (batas 1,5 × IQR).
   - Standardisasi fitur numerik dengan `StandardScaler`.
   - Menghapus baris duplikat.
   - Encoding kategorikal dengan One-Hot Encoding dan Label Encoding.
3. **EDA.** Histogram distribusi, matriks korelasi, dan korelasi fitur terhadap target `Class`.
4. **Data splitting.** `train_test_split` dengan `test_size=0.2` dan `random_state=1`. Dataset Wine Quality diproses dan dibagi dengan cara yang sama.
5. **Pelatihan model.**
   - `Lars(n_nonzero_coefs=1)`
   - `LinearRegression()`
   - `GradientBoostingRegressor(random_state=184)`
6. **Evaluasi.** Menghitung MAE, MSE, dan R² pada data uji, lalu merangkumnya dalam satu `DataFrame`.

---

## 4. Hasil Evaluasi

Notebook tidak menyimpan output sel, sehingga angka di bawah perlu diisi setelah notebook dijalankan ulang (`Runtime > Run all`).

| Model | MAE ↓ | MSE ↓ | R² ↑ |
|---|---|---|---|
| Lars | _isi_ | _isi_ | _isi_ |
| Linear Regression | _isi_ | _isi_ | _isi_ |
| Gradient Boosting Regressor | _isi_ | _isi_ | _isi_ |

Keterangan: ↓ berarti semakin kecil semakin baik, ↑ berarti semakin besar (mendekati 1) semakin baik.

**Cara menentukan model terbaik:** pilih model dengan MAE dan MSE paling rendah serta R² paling mendekati 1. Jika ketiganya tidak sejalan, jelaskan trade-off-nya dalam pembahasan.

---

## 5. Cara Menjalankan

```bash
git clone <url-repo-anda>
cd <nama-repo>
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
jupyter notebook projectuts.ipynb
```

Di Google Colab, unggah `Rice_Cammeo_Osmancik.csv` dan `winequality.csv` ke `/content/` lalu jalankan semua sel.

---

## 6. Catatan Metodologi

Beberapa hal berikut sebaiknya diketahui dan bisa dibahas sebagai keterbatasan:

- **Target biner pada model regresi.** `Class` hanya bernilai 0 dan 1, padahal model yang dipakai adalah regresor. Untuk target seperti ini, klasifikasi (misalnya Logistic Regression) dengan metrik accuracy, precision, recall, dan F1 biasanya lebih tepat. Hasil regresi di sini tetap valid sebagai latihan evaluasi MAE, MSE, dan R².
- **Urutan preprocessing.** Standardisasi dilakukan sebelum pembagian data, sehingga statistik data uji ikut terpakai. Praktik yang lebih aman adalah `fit` scaler pada data latih saja (*data leakage*).
- **Penghapusan outlier.** Baris outlier dibuang berdasarkan IQR sebelum winsorizing kolom `Area`, sehingga langkah winsorizing praktis tidak berpengaruh pada baris yang tersisa.
- **Dataset Wine Quality.** Pada notebook, dataset ini baru sampai tahap pemrosesan dan pembagian data. Model belum dilatih dan dievaluasi.
- **`n_nonzero_coefs=1` pada LARS** membatasi model hanya memakai satu fitur, sehingga wajar jika kinerjanya lebih rendah dibanding model lain.

---

## 7. Referensi

Peringkat SINTA mengacu pada informasi di situs jurnal masing-masing. Verifikasi ulang di [SINTA](https://sinta.kemdiktisaintek.go.id) sebelum dikumpulkan.

### 7.1 Jurnal nasional terakreditasi SINTA 3 ke atas (terbit 2022 sampai 2023)

| No | Referensi | SINTA | Relevansi dengan proyek |
|---|---|---|---|
| 1 | Pardede, J., & Rayyan, R. (2023). House Prices Prediction: Multiple Linear Regression vs Ridge vs Polynomial. *MIND (Multimedia Artificial Intelligent Networking Database) Journal*, 8(1), 14–26. https://doi.org/10.26760/mindjournal.v8i1.14-26 | SINTA 3 | Membandingkan tiga model regresi dengan MSE, RMSE, dan MAE. Model dengan nilai error terkecil dinyatakan terbaik |
| 2 | Haris, M. S., Anshori, M., & Khudori, A. N. (2023). Prediction of Stunting Prevalence in East Java Province with Random Forest Algorithm. *Jurnal Teknik Informatika (JUTIF)*, 4(1), 11–13. https://doi.org/10.52436/1.jutif.2023.4.1.614 | SINTA 3 pada periode terbit (2020 sampai Vol 5 No 2 2024), kini SINTA 2 | Evaluasi dengan MAE dan MSE. Model dengan MAE dan MSE lebih kecil (regresi linear berganda) dinilai lebih baik daripada Random Forest |
| 3 | Haris, M. S., Khudori, A. N., & Kusuma, W. T. (2022). Perbandingan Metode Supervised Machine Learning untuk Prediksi Prevalensi Stunting di Provinsi Jawa Timur. *Jurnal Teknologi Informasi dan Ilmu Komputer (JTIIK)*, 9(7), 1571–1576. https://doi.org/10.25126/jtiik.2022976744 | Terakreditasi Kemenristekdikti (SK 36/E/KPT/2019) | Membandingkan Linear Regression, SVR, dan Random Forest dengan MAE dan MSE |
| 4 | Sholeh, M., Suraya, S., & Andayati, D. (2022). Machine Linear untuk Analisis Regresi Linier Biaya Asuransi Kesehatan dengan Menggunakan Python Jupyter Notebook. *JEPIN (Jurnal Edukasi dan Penelitian Informatika)*, 8(1), 20–27. | Terakreditasi (tercatat SINTA 2 pada dokumen penilaian 2020) | Regresi linear di Python dengan koefisien determinasi (R²) dan MSE, alur kerja mirip notebook ini |
| 5 | Nurani, A. T., dkk. (2023). Perbandingan Kinerja Regresi Decision Tree dan Regresi Linear Berganda. *Jurnal Sains dan Edukasi Sains (JuSES)*, 6(1), 34–43. https://ejournal.uksw.edu/juses/article/download/8438/2430/36568 | SINTA 3 saat ini (SINTA 4 pada saat terbit) | Membahas MAE, MAPE, RMSE, dan R² beserta cara membacanya |

### 7.2 Referensi pendukung (internasional)

6. Chicco, D., Warrens, M. J., & Jurman, G. (2021). The coefficient of determination R-squared is more informative than SMAPE, MAE, MAPE, MSE and RMSE in regression analysis evaluation. *PeerJ Computer Science*, 7, e623. https://doi.org/10.7717/peerj-cs.623

### 7.3 Dokumentasi

7. Scikit-learn. *Metrics and scoring: `mean_absolute_error`, `mean_squared_error`, `r2_score`.* https://scikit-learn.org/stable/modules/model_evaluation.html

---

## Lisensi

Proyek ini dibuat untuk keperluan akademik (UTS).
