# Laporan Proyek Machine Learning - Wildan Andika Permana

## Domain Proyek

### Latar belakang masalah:
Dalam lingkungan ritel yang sangat kompetitif saat ini, peningkatan kualitas layanan merupakan hal yang paling penting untuk menjamin kelangsungan gerai ritel dan memenuhi harapan pelanggan. Data yang diangkut dan dikelola dengan benar tidak hanya meningkatkan kepuasan pelanggan tetapi juga berkontribusi terhadap profitabilitas. Pengecer sangat bergantung pada data pelanggan untuk memahami tren perilaku, menggunakan metrik seperti tingkat konversi, persen nilai pesanan, riwayat transaksi terkini untuk mengidentifikasi perilaku pelanggan, namun tantangannya adalah data ini akan digunakan secara efektif untuk menghasilkan intelijen yang dapat ditindaklanjuti dan menginformasikan keputusan strategis.

### Mengapa dan bagaimana mengatasi masalah tersebut:
Integrasi analisis profitabilitas pelanggan (CPA) dapat mengubah strategi pemasaran di ritel. Dengan memperoleh wawasan tentang profitabilitas pelanggan individu dan memahami distribusi keuntungan di seluruh pelanggan, tenaga penjualan dapat membuat keputusan yang tepat mengenai biaya, pendapatan, mitigasi risiko, dan posisi pasar. CPA membantu memisahkan pelanggan berdasarkan profitabilitas, mengidentifikasi segmen "pelanggan bernilai tinggi" yang menghasilkan keuntungan signifikan. Sehingga dapat menimbulkan promosi yang paling efektif untuk setiap segmen pelanggan, memaksimalkan ROI dan meminimalkan pemborosan sumber daya membuat tenaga penjualan dapat menyesuaikan strategi pemasaran untuk menargetkan pelanggan bernilai tinggi dengan lebih baik.

Sebagai contoh saat ini, perusahaan memiliki banyak sekali data pelanggan. Sumbernya bisa dari toko online, media sosial, atau sekadar transaksi penjualan. Dari data ini, dapat memperoleh pola dan korelasi. Dan analisis prediktif dapat membuat perkiraan tentang perilaku pelanggan di masa depan. Di sektor B2B, ini adalah tujuan perusahaan, seperti pertumbuhan, peningkatan penjualan, atau posisi kompetitif yang baik. Namun, ekspektasi pelanggan seringkali bersifat individual dan berhubungan dengan layanan atau produk tertentu. Harapan-harapan ini terus berubah dan oleh karena itu harus ditentukan secara berkelanjutan.

## Business Understanding

### Problem Statements
Berdasarkan uraian yang telah dipaparkan pada latar belakang diatas, maka dapat diambil sebuah rumusan masalah yang dirumuskan sebagai berikut:

- Bagaimana mengoptimalkan proses bisnis dan layanan untuk memaksimalkan kepuasan pelanggan berdasarkan prediksi pola dan korelasi di masa depan?
- Bagaimana mengelola dan memanfaatkan data pelanggan untuk meningkatkan kepuasan dan mendorong profitabilitas, dengan mempertimbangkan prediksi pola dan korelasi pelanggan di masa depan?
- Bagaimana cara membuat model prediksi dengan akurasi yang baik?

### Goals
Berdasarkan rumusan masalah yang telah dipaparkan di atas, maka proyek penelitian ini memiliki tujuan, yaitu:

- Mengetahui model algoritma machine learning dalam memanfaatkan data pelanggan secara efektif untuk mengetahui hubungan antara optimalisasi proses bisnis pelayanan dan ekspektasi pelanggan.
- Mengetahui model pengembangan model machine learning dalam optimalisasi proses bisnis untuk memaksimalkan efisiensi dan profitabilitas.
- Mengetahui hasil evaluasi model machine learning dalam membantu para peritel memanfaatkan data untuk pengambilan keputusan strategis.

### Solution statements
Berdasarkan tujuan yang telah dipaparkan diatas, maka proyek penelitian ini memiliki solusi atau tahapan sebagai berikut:

1. Analisis Profitabilitas Pelanggan (CPA)
  - Menganalisis metrik profitabilitas individual seperti jumlah pembelian, frekuensi pembelian, dan harga pembelian rata-rata.
  - Mengelompokkan pelanggan berdasarkan tingkat profitabilitas.
  - Mengevaluasi perubahan tingkat retensi pelanggan, nilai pembelian rata-rata, dan pertumbuhan pendapatan secara keseluruhan.
    
2. Teknik Pemodelan Menggunakan Algoritma Klasifikasi
  - Mengidentifikasi kemacetan, mengoptimalkan alokasi sumber daya, dan meningkatkan efisiensi alur kerja.
  - Mengevaluasi pengurangan waktu siklus, biaya penyimpanan inventaris, dan biaya operasional.
  - Algoritma yang akan dipakai diantaranya adalah sebagai berikut:
  - K-Nearest Neighbors (KNN)
  - Adaptive Boosting (Adaboost)
  - Random Forest
  - Support Vector Machine (SVM)

3. Evaluasi Model Analisis Prediktif untuk Perkiraan Permintaan
  - Memperkirakan permintaan produk di masa depan.
  - Mengoptimalkan manajemen inventaris dan meningkatkan efisiensi rantai pasokan.
  - Mengevaluasi keakuratan perkiraan, tingkat perputaran inventaris, dan pengurangan kehabisan stok.

## Data Understanding
Dataset yang dipakai dalam proyek machine learning ini merupakan dataset berjudul Supermarket sales. Dataset ini dipubilkasikan oleh AUNG PYAE melalui platform Kaggle. Dataset ini terdiri dari file berformat csv (comma-separated values) dengan ukuran total 131.53 kB.

Link: [Supermarket sales](https://www.kaggle.com/datasets/aungpyaeap/supermarket-sales).

**_Tabel 1. Tampilan dataset dalam bentuk dataframe dengan pandas_**

|   | Invoice ID |   Branch  |   City  | Customer type | Gender |   Product line  |   Unit price   |   Quantity   |   Tax 5%   |   Total   |   Date   |   Time   |   Payment   |   cogs   |   gross margin percentage   |   gross income   |   Rating   |
|---|:---:|:------:|:------:|:--------:|:------:|:---------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|
| 0 | 750-67-8428 | A | Yangon | Member | Female | Health and beauty | 74.69 | 7 | 26.1415 | 548.9715 | 1/5/2019 | 13:08 | Ewallet | 522.83 | 4.761904762 | 26.1415 | 9.1 |
| 1 | 226-31-3081 | C | Naypyitaw | Normal | Female | Electronic accessories | 15.28 | 5 | 3.82 | 80.22 | 3/8/2019 | 10:29 | Cash | 76.4 | 4.761904762 | 3.82 | 9.6 |
| 2 | 631-41-3108 | A | Yangon | Normal | Male | Home and lifestyle | 46.33 | 7 | 16.2155 | 340.5255 | 3/3/2019 | 13:23 | Credit card | 324.31 | 4.761904762 | 16.2155 | 7.4 |
| 3 | 123-19-1176 | A | Yangon | Member | Male | Health and beauty | 58.22 | 8 | 23.288 | 489.048 | 1/27/2019 | 20:33 | Ewallet | 465.76 | 4.761904762 | 23.288 | 8.4 |
| 4 | 373-73-7910 | A | Yangon | Normal | Male | Sports and travel | 86.31 | 7 | 30.2085 | 634.3785 | 2/8/2019 | 10:37 | Ewallet | 604.17 | 4.761904762 | 30.2085 | 5.3 |

### Variabel-variabel pada Retail Case Study Data adalah sebagai berikut:
- Invoice id: Computer generated sales slip invoice identification number
- Branch: Branch of supercenter (3 branches are available identified by A, B and C).
- City: Location of supercenters
- Customer type: Type of customers, recorded by Members for customers using member card and Normal for without member card.
- Gender: Gender type of customer
- Product line: General item categorization groups - Electronic accessories, Fashion accessories, Food and beverages, Health and beauty, Home and lifestyle, Sports and travel
- Unit price: Price of each product in $
- Quantity: Number of products purchased by customer
- Tax: 5% tax fee for customer buying
- Total: Total price including tax
- Date: Date of purchase (Record available from January 2019 to March 2019)
- Time: Purchase time (10am to 9pm)
- Payment: Payment used by customer for purchase (3 methods are available – Cash, Credit card and Ewallet)
- COGS: Cost of goods sold
- Gross margin percentage: Gross margin percentage
- Gross income: Gross income
- Rating: Customer stratification rating on their overall shopping experience (On a scale of 1 to 10)

![Informasi dari dataset](https://github.com/Wildanae123/Machine-Learning-Terapan-Predictive-Analytics/blob/main/images/Informasi%20dari%20dataset.png)

**_Gambar 1. Informasi dari dataset_**

Berdasarkan pada gambar di atas, dapat diketahui bahwa :
* Terdapat 4 kolom dengan tipe objek yaitu : Customer type, Gender, Product line, dan Payment. kolom ini merupakan  _categorical features_ (fitur non-numerik).
* Terdapat 1 kolom bertipe numerik dengan tipe data int64 yaitu Quantity.
* Terdapat 3 kolom bertipe numerik dengan tipe data float64 yaitu Unit price, Total dan Rating. Kolom Total akan dijadikan kolom target pada proyek ini.

### Visualisasi data atau Exploratory Data Analysis
#### Outliers
Adalah sampel yang nilainya sangat jauh dari cakupan umum data utama. outliers sendiri adalah hasil pengamatan yang kemunculannya sangat jarang dan berbeda dari data hasil pengamatan lainnya

Ada beberapa teknik untuk menangani outliers, antara lain:
- Hypothesis Testing
- Z-score method
- IQR Method

Untuk mengecek apakah ada outliers atau tidak, dapat menggunakan teknik visualisasi boxplot. Dengan boxplot ukuran lokasi dan penyebaran, serta informasi tentang simetri dan outliers bisa digambarkan secara vertikal maupun horizontal.

![Visualisasi boxplot](https://github.com/Wildanae123/Machine-Learning-Terapan-Predictive-Analytics/blob/main/images/Visualisasi%20boxplot.png)

**_Gambar 2. Visualisasi boxplot_**

Penanganan Outliers
Pada proyek ini , terdapat outliers variabel total, untuk menangani outliers kita dapat menggunakan teknik IQR method, IQR adalah singkatan dari Interquartile Range.
Menggunakan persamaan berikut:

Batas bawah = Q1 - 1.5 * IQR

Batas atas = Q3 + 1.5 * IQR

Menghasilkan output sebagai berikut:

![Outliner](https://github.com/Wildanae123/Machine-Learning-Terapan-Predictive-Analytics/blob/main/images/Outliner.png)

**_Gambar 3. Outliner_**

#### Memahami data dengan statistics

![Deskripsi Variabel](https://github.com/Wildanae123/Machine-Learning-Terapan-Predictive-Analytics/blob/main/images/Deskripsi%20Variabel.png)

**_Gambar 4. Deskripsi Variabel_**

Fungsi describe() memberikan informasi statistik pada masing-masing kolom, antara lain:
- Count  adalah jumlah sampel pada data.
- Mean adalah nilai rata-rata.
- Std adalah standar deviasi.
- Min yaitu nilai minimum setiap kolom. 
- 25% adalah kuartil pertama. Kuartil adalah nilai yang menandai batas interval dalam empat bagian sebaran yang sama. 
- 50% adalah kuartil kedua, atau biasa juga disebut median (nilai tengah).
- 75% adalah kuartil ketiga.
- Max adalah nilai maksimum.

#### Memahami data dengan visualization menggunakan teknik Univariate Analysis
##### Categorical Features

![Univariate Analysis Categorical Features](https://github.com/Wildanae123/Machine-Learning-Terapan-Predictive-Analytics/blob/main/images/Univariate%20Analysis%20Categorical%20Features.png)

**_Gambar 5. Univariate Analysis Categorical Features_**

Berdasarkan deskripsi variabel di atas, kita bisa memperoleh beberapa informasi, antara lain:
- Kategori branch dari yang paling banyak ke yang paling sedikit adalah A, B, dan C.
- Kategori customer type dari yang paling banyak ke yang paling sedikit adalah member dan kemudian normal.
- Kategori gender dari yang paling banyak ke yang paling sedikit adalah female dan kemudian male.
- Kategori product line dari yang paling banyak ke yang paling sedikit adalah Fashion accessories, Food and beverages, Electronic accessories, Sports and travel, Home and lifestyle, dan Health and beauty.
- Kategori paymenet dari yang paling banyak ke yang paling sedikit adalah Ewallet, Cash, dan Credit card.

##### Numerical Features

![Univariate Analysis Numerical Features](https://github.com/Wildanae123/Machine-Learning-Terapan-Predictive-Analytics/blob/main/images/Univariate%20Analysis%20Numerical%20Features.png)

**_Gambar 6. Univariate Analysis Numerical Features_**

Mari amati histogram di atas, khususnya histogram untuk variabel "Total" yang merupakan fitur target (label) pada data kita, kita bisa memperoleh beberapa informasi, antara lain:

- Peningkatan harga barang sebanding dengan penurunan jumlah sampel, Hal ini dapat kita lihat jelas dari histogram "Total" yang grafiknya mengalami penurunan seiring dengan semakin banyaknya jumlah sampel (sumbu x).
- Rentang harga barang yang dibeli per pelanggan cukup tinggi yaitu dari skala 0 hingga 1000
- Setengah harga barang dibeli di bawah harga 200.
- Distribusi Total miring ke kanan (right-skewed). Hal ini akan berimplikasi pada model.

#### Memahami data dengan visualization menggunakan teknik Multivariate Analysis
##### Categorical Features

![Multivariate Analysis Categorical Features](https://github.com/Wildanae123/Machine-Learning-Terapan-Predictive-Analytics/blob/main/images/Multivariate%20Analysis%20Categorical%20Features.png)

**_Gambar 7. Multivariate Analysis Categorical Features_**

Dengan mengamati rata-rata Total relatif terhadap fitur kategori di atas, kita memperoleh insight sebagai berikut:

- Pada fitur ‘branch’, rata rata Total cenderung mirip rentangnya berada antara 5 hingga 6. grade tertinggi yaitu grade ideal memiliki harga rata-rata terendah diantara grade lainnya. sehingga, fitur **branch** memiliki pengaruh atau dampak yang kecil terhadap rata-rata Total.
- Pada fitur ‘customer type’, rata rata Total cenderung mirip rentangnya berada antara 5 hingga 6. grade tertinggi yaitu grade ideal memiliki harga rata-rata terendah diantara grade lainnya. sehingga, fitur **customer type** memiliki pengaruh atau dampak yang kecil terhadap rata-rata Total.
- Pada fitur ‘gender’, rata rata Total memiliki perbedaan yang cukup signifikan terhadap female di rentang 6 hingga 7 dengan male yang hanya berada di rentang 5 hingga 6, sehingga fitur **gender** memiliki pengaruh atau dampak yang besar terhadap rata rata Total.
- Pada fitur ‘product line’, rata rata Total cenderung mirip rentangnya berada antara 5 hingga 6. grade tertinggi yaitu grade ideal memiliki harga rata-rata terendah diantara grade lainnya. sehingga, fitur **product line** memiliki pengaruh atau dampak yang kecil terhadap rata-rata Total.
- Pada fitur ‘payment’, rata rata Total cenderung mirip rentangnya berada antara 5 hingga 6. grade tertinggi yaitu grade ideal memiliki harga rata-rata terendah diantara grade lainnya. sehingga, fitur **payment** memiliki pengaruh atau dampak yang kecil terhadap rata-rata Total.
- Kesimpulan akhir, fitur kategori memiliki pengaruh yang rendah terhadap Total pada Fitur branch, customer type, product line, dan payment tetapi memiliki pengaruh tinggi terhadap **gender**
  
##### Numerical Features

![Multivariate Analysis Numerical Features](https://github.com/Wildanae123/Machine-Learning-Terapan-Predictive-Analytics/blob/main/images/Multivariate%20Analysis%20Numerical%20Features.png)

**_Gambar 8. Multivariate Analysis Numerical Features_**

Pada kasus ini, kita akan melihat relasi antara semua fitur numerik dengan fitur target kita yaitu ‘Total’. Untuk membacanya, perhatikan fitur pada sumbu y, temukan fitur target ‘Total’, dan lihatlah grafik relasi antara semua fitur pada sumbu x dengan fitur price pada sumbu y. Dalam hal ini, kita cukup melihat relasi antar fitur numerik dengan fitur target ‘Total’ pada baris tersebut saja.

Pada pola sebaran data grafik pairplot sebelumnya, terlihat 'Quantity' memiliki korelasi yang tinggi dengan fitur 'Total'. Sedangkan fitur lainnya yaitu 'Rating' terlihat memiliki korelasi yang lemah karena sebarannya tidak membentuk pola. Untuk mengevaluasi skor korelasinya, gunakan fungsi corr().

![/assets/images/Correlation Matriks.png](https://github.com/Wildanae123/Machine-Learning-Terapan-Predictive-Analytics/blob/main/images/Correlation%20Matriks.png)

**_Gambar 9. Correlation Matriks_**

Pada grafik korelasi di atas. Jika kita amati, fitur 'Quantity' memiliki skor korelasi yang besar (di atas 0.1) dengan fitur target 'Total'. Artinya, fitur 'Quantity' berkorelasi tinggi dengan keempat fitur tersebut. Sementara itu, fitur 'Rating' memiliki korelasi yang sangat kecil (-0.04). Sehingga, fitur tersebut dapat di-drop.

## Data Preparation
Pada bagian ini penerapan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.
- Melakukan proses Encoding menggunakan teknik One-Hot Encoding, proses encoding sangat diperlukan, agar data yang masuk ke dalam algoritma machine learning dapat bekerja dengan baik. Sebagian besar algoritma klasifikasi lebih mudah untuk memproses nilai numerical daripada kategorikal.
Teknik One-Hot Encoding dipilih untuk alasan berikut:

  - Kesederhanaan: Teknik ini mudah dipahami dan diimplementasikan.
  - Efisiensi: Teknik ini efisien dalam hal memori dan waktu komputasi.
  - Kompatibilitas: Teknik ini kompatibel dengan berbagai algoritma machine learning.
  - Kemampuan Interpretability: Teknik ini menghasilkan data yang mudah diinterpretasikan.

  **_Tabel 2. Hasil Encoding_**

  ![Hasil Encoding](https://github.com/Wildanae123/Machine-Learning-Terapan-Predictive-Analytics/blob/main/images/Hasil%20Encoding.png)
  
- Melakukan Reduksi menggunakan teknik PCA, diperlukan untuk mereduksi variabel asli menjadi sejumlah kecil variabel baru yang tidak berkorelasi linier, disebut komponen utama (PC). Komponen utama ini dapat menangkap sebagian besar varians dalam variabel asli. Sehingga, saat teknik PCA diterapkan pada data, ia hanya akan menggunakan komponen utama dan mengabaikan sisanya.
Teknik PCA dipilih karena:

  - Efektivitas: Teknik ini efektif dalam mengurangi dimensi data tanpa kehilangan informasi signifikan.
  - Kemampuan Interpretability: Teknik ini menghasilkan komponen utama yang mudah diinterpretasikan.
  - Kestabilan: Teknik ini stabil dan tidak sensitif terhadap outlier.

  ![Hasil Reduksi](https://github.com/Wildanae123/Machine-Learning-Terapan-Predictive-Analytics/blob/main/images/Hasil%20Reduksi.png)

  **_Gambar 10. Hasil Reduksi_**
  
- Melakukan Data Splitting, membagi dataset menjadi data latih (train) dan data uji (test) merupakan hal yang harus kita lakukan sebelum membuat model. Kita perlu mempertahankan sebagian data yang ada untuk menguji seberapa baik generalisasi model terhadap data baru. Ketahuilah bahwa setiap transformasi yang kita lakukan pada data juga merupakan bagian dari model. Karena data uji (test set) berperan sebagai data baru, kita perlu melakukan semua proses transformasi dalam data latih. Inilah alasan mengapa langkah awal adalah membagi dataset sebelum melakukan transformasi apa pun. Tujuannya adalah agar kita tidak mengotori data uji dengan informasi yang kita dapat dari data latih dengan parameter yang digunakan yaitu :

  - X berfungsi untuk menghapus kolom charges
  - y berfungsi menampilkan kolom charges
  - test_size adalah ukuran pembagian dataset yaitu sekitar 80 % untuk training dan 20 % untuk testing, data testing ini bertujuan untuk mengukur kinerja model pada data baru.
  - random_state: digunakan untuk mengontrol random number generator yang digunakan, di proyek ini menggunakan random_state = 123
  
  ![Hasil Data Splitting](https://github.com/Wildanae123/Machine-Learning-Terapan-Predictive-Analytics/blob/main/images/Hasil%20Data%20Splitting.png)

  **_Gambar 11. Hasil Data Splitting_**

- Melakukan Standarisasi, Standarisasi adalah teknik transformasi yang paling umum digunakan dalam tahap persiapan pemodelan. Pada proyek ini, scaling data menggunakan metode standarisasi dengan teknik StandarScaler dari library Scikit learn, karena secara umum distribusi data berada pada kondisi normal dan standarisasi lebih cocok untuk digunakan dalam case yang seperti ini. StandardScaler menghasilkan distribusi dengan standar deviasi sama dengan 1 dan mean sama dengan 0.
Metode StandarScaler dipilih karena:

  - Kesesuaian dengan Distribusi Data: Teknik ini cocok untuk data yang terdistribusi normal.
  - Kemampuan Scaling: Teknik ini menghasilkan data dengan standar deviasi 1 dan mean 0, yang membantu meningkatkan performa algoritma machine learning.
  - Kestabilan: Teknik ini stabil dan tidak sensitif terhadap outlier.

  ![Hasil Standarisasi](https://github.com/Wildanae123/Machine-Learning-Terapan-Predictive-Analytics/blob/main/images/Hasil%20Standarisasi.png)

  **_Gambar 12. Hasil Standarisasi_**

## Modeling
Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan.

> [!NOTE]
> Pada proyek ini, algoritma machine learning yang dipakai adalah K-Nearest Neighbor, Random Forest, Boosting Algorithm dan Support Vector Machine.

**Model Development dengan K-Nearest Neighbor**
KNN bekerja dengan membandingkan jarak satu sampel ke sampel pelatihan lain dengan memilih sejumlah k tetangga terdekat (dengan k adalah sebuah angka positif). Nah, itulah mengapa algoritma ini dinamakan K-nearest neighbor (sejumlah k tetangga terdekat). KNN bisa digunakan untuk kasus klasifikasi dan regresi.

**Kelebihan & Kekurangan KNN**

`Kelebihan`
- Mudah dipahami dan diimplementasikan.
- Berkinerja baik pada data dengan dimensi rendah.
- Fleksibel dan dapat digunakan untuk klasifikasi dan regresi.

`Kekurangan`
- Sensitif terhadap outlier dan noise.
- Kinerja menurun pada data dengan dimensi tinggi.
- Membutuhkan waktu komputasi yang lama untuk data yang besar.

**Model Development dengan Random Forest**
Algoritma random forest adalah salah satu algoritma supervised learning. Ia dapat digunakan untuk menyelesaikan masalah klasifikasi dan regresi. Random forest juga merupakan algoritma yang sering digunakan karena cukup sederhana tetapi memiliki stabilitas yang mumpuni.

**Kelebihan & Kekurangan Random Forest**

`Kelebihan`
- Akurat dan robust.
- Mampu menangani data dengan dimensi tinggi.
- Dapat mengatasi overfitting.

`Kekurangan`
- Interpretasi model yang kompleks.
- Membutuhkan waktu training yang lama.
- Sensitif terhadap hyperparameter tuning.

**Model Development dengan Boosting Algorithm**
Algoritma yang menggunakan teknik boosting bekerja dengan membangun model dari data latih. Kemudian ia membuat model kedua yang bertugas memperbaiki kesalahan dari model pertama. Model ditambahkan sampai data latih terprediksi dengan baik atau telah mencapai jumlah maksimum model untuk ditambahkan. Pada kasus ini, kita akan menggunakan metode adaptive boosting.

**Kelebihan & Kekurangan AdaBOOST**

`Kelebihan`
- Akurat dan dapat menangani data yang kompleks.
- Mampu mengatasi overfitting.
- Berkinerja baik pada data dengan noise.

`Kekurangan`
- Interpretasi model yang kompleks.
- Sensitif terhadap outlier.
- Membutuhkan waktu training yang lama.

**Model Development dengan Support Vector Machine (SVM)**
Tujuan dari algoritma SVM adalah untuk menemukan hyperplane terbaik dalam ruang berdimensi-N (ruang dengan N-jumlah fitur) yang berfungsi sebagai pemisah yang jelas bagi titik-titik data input.

**Kelebihan & Kekurangan SVM**

`Kelebihan`
- Akurat dan robust.
- Mampu menangani data dengan dimensi tinggi.
- Efisien dalam penggunaan memori.
 
`Kekurangan`
- Interpretasi model yang kompleks.
Membutuhkan waktu training yang lama untuk data yang besar.
Sensitif terhadap outlier dan noise.

**Pemilihan Model::**
Pemilihan parameter yang optimal untuk setiap algoritma sangat penting untuk mencapai performa terbaik. Dari keempat model yang telah dilatih, terlihat bahwa prediksi dengan K-Nearest Neighbor memberikan hasil yang paling mendekati. Sehingga **K-Nearest Neighbor** merupakan model terbaik yang dihasilkan. Hal ini didasarkan pada hasil uji pada gambar berikut:

![Visualisasi bar chart MSE](https://github.com/Wildanae123/Machine-Learning-Terapan-Predictive-Analytics/blob/main/images/Visualisasi%20bar%20chart%20MSE.png)

**_Gambar 13. Visualisasi bar chart MSE_**

## Evaluation
Sebelum model diterapkan, model perlu dievaluasi agar terbukti cocok untuk tujuan yang telah ditentukan. Fase ini bertujuan untuk memastikan bahwa model akan mampu membuat prediksi yang akurat dan tidak mengalami overfitting atau underfitting.

Mengevaluasi model regresi sebenarnya relatif sederhana. Secara umum, hampir semua metrik adalah sama. Jika prediksi mendekati nilai sebenarnya, performanya baik. Sedangkan jika tidak, performanya buruk. Secara teknis, selisih antara nilai sebenarnya dan nilai prediksi disebut eror. Maka, semua metrik mengukur seberapa kecil nilai eror tersebut.

Metrik yang akan kita gunakan pada prediksi ini adalah MSE atau Mean Squared Error yang menghitung jumlah selisih kuadrat rata-rata nilai sebenarnya dengan nilai prediksi. Mean Squared Error (MSE) adalah metrik yang digunakan untuk mengukur ketepatan model regresi dalam memprediksi nilai target. MSE menghitung rata-rata dari kuadrat perbedaan antara nilai prediksi dan nilai target yang sebenarnya.

Nilai MSE memiliki beberapa signifikansi, antara lain:
- Akurasi Prediksi: Nilai MSE yang rendah menunjukkan bahwa model mampu memprediksi nilai target dengan lebih akurat.
- Kecocokan Model: Nilai MSE yang rendah menunjukkan bahwa model cocok dengan data yang digunakan.
- Perbandingan Model: Nilai MSE dapat digunakan untuk membandingkan kinerja model yang berbeda.

MSE didefinisikan dalam persamaan berikut

![Perhitungan MSE](https://github.com/Wildanae123/Machine-Learning-Terapan-Predictive-Analytics/blob/main/images/Perhitungan%20MSE.jpeg)

**_Gambar 14. Perhitungan MSE_**

_Keterangan:_

_N = jumlah dataset_

_yi = nilai sebenarnya_

_y_pred = nilai prediksi_

Hasil evaluasi pada data latih dan data test adalah sebagai berikut.

**_Tabel 3. Hasil Perhitungan MSE 4 Algoritma_**
| | train | test |
| ------------- | ------------- | ------------- |
| KNN | 4.751774 | 6.166513 |
| RF | 0.013235 | 0.081261 |
| Boosting | 3.795002 | 4.343941 |
| SVM | 49.703125 | 46.693444 |

Untuk memudahkan, mari kita plot matrik tersebut dengan bar chart

![Visualisasi bar chart MSE](https://github.com/Wildanae123/Machine-Learning-Terapan-Predictive-Analytics/blob/main/images/Visualisasi%20bar%20chart%20MSE.png)

**_Gambar 15. Visualisasi bar chart MSE_**

Dari gambar di atas , terlihat bahwa, model Random Forest (RF) memiliki nilai error pada data test yang paling kecil sedangkan model Support Vector Machine memiliki nilai error paling banyak dibandingkan dari ketiga model. Hal ini menunjukkan bahwa RF mampu memprediksi nilai target dengan lebih akurat dibandingkan dengan model lain.

Untuk mengujinya, mari kita buat prediksi menggunakan beberapa harga dari data test.

**_Tabel 4. Hasil Prediksi MSE_**
| x | y_true	 | prediksi_KNN | prediksi_RF | prediksi_Boosting | prediksi_SVM |
| --- | --- | --- | --- | --- | --- |
| 131 | 580.419 | 581.1 | 582.2 | 573.0 | 303.2 |

Pada Tabel 4 adalah hasil prediksi "Total" dari 4 algoritma yaitu K-Nearest Neighbor, Random Forest, AdaBOOST dan Support Vector Machine. Terlihat bahwa prediksi dengan Random Forest(RF) dan K-Nearest Neighbor memberikan hasil yang paling mendekati. Dimana algoritma K-Nearest Neighbor memiliki nilai prediksi MSE (Mean Squared Error) sebesar 581.1, algoritma Random Forest memiliki nilai prediksi MSE (Mean Squared Error) sebesar 582.2, algoritma AdaBOOST memiliki nilai prediksi MSE (Mean Squared Error) sebesar 573.0 sedangkan algoritma Support Vector Machine memiliki nilai prediksi MSE (Mean Squared Error) sebesar 303.2.

**Kesimpulan**

Dari Empat model Algoritma yang dikembangkan berdasarkan hasil perbandingan dan visualisasi, dapat disimpulkan bahwa model Random Forest merupakan model terbaik dengan nilai error terkecil pada data test. Model K-Nearest Neighbor memberikan hasil yang mendekati, namun RF lebih unggul dalam hal akurasi.

Beberapa faktor yang dapat menyebabkan perbedaan antara nilai prediksi dan nilai sebenarnya, antara lain:
- Kompleksitas model: Model yang terlalu kompleks dapat overfit data dan menghasilkan prediksi yang tidak akurat.
- Kualitas data: Data yang tidak akurat atau tidak lengkap dapat menyebabkan prediksi yang tidak akurat.
- Variabilitas data: Data yang sangat bervariasi dapat membuat model sulit untuk memprediksi nilai target dengan tepat.
- Algoritma yang digunakan: Algoritma yang berbeda memiliki kekuatan dan kelemahannya masing-masing, dan beberapa algoritma mungkin lebih cocok untuk data tertentu daripada yang lain.

**Saran**

- Melakukan tuning hyperparameter untuk meningkatkan performa model.
- Mencoba model regresi lainnya untuk dibandingkan dengan model yang telah diuji.

**---Ini adalah bagian akhir laporan---**

**Referensi:**

[1] [van Raaij, E.M. (2005), "The strategic value of customer profitability analysis", Marketing Intelligence & Planning, Vol. 23 No. 4, pp. 372-381.](https://doi.org/10.1108/02634500510603474)

[2] [K. Vergidis, A. Tiwari and B. Majeed, "Business Process Analysis and Optimization: Beyond Reengineering," in IEEE Transactions on Systems, Man, and Cybernetics, Part C (Applications and Reviews), vol. 38, no. 1, pp. 69-82, Jan. 2008.](https://doi.org/10.1109/TSMCC.2007.905812)

[3] [Krumeich, J., Werth, D. & Loos, P. Prescriptive Control of Business Processes. Bus Inf Syst Eng 58, 261–280 (2016).](https://doi.org/10.1007/s12599-015-0412-2)
