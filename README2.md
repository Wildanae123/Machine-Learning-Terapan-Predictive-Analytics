# Laporan Proyek Machine Learning - Wildan Andika Permana

## Domain Proyek

### Latar belakang masalah:
Dunia film telah menjadi lanskap yang luas dan dinamis, dipenuhi dengan ribuan judul baru bermunculan setiap tahun. Bagi pencinta film, kelimpahan ini justru bisa menjadi tantangan. Dengan begitu banyak pilihan, menemukan film yang tepat untuk ditonton bisa terasa seperti mencari jarum dalam tumpukan jerami. Faktor-faktor seperti genre, aktor, sutradara, dan ulasan kritikus semuanya ikut berperan dalam keputusan menonton, namun seringkali prosesnya menjadi membingungkan dan memakan waktu.

Pengguna yang tidak puas dengan rekomendasi yang diberikan cenderung kehilangan minat dan berhenti menggunakan platform film. Yang akhirnya menyebabkan Platform film kehilangan potensi pendapatan ketika pengguna tidak dapat menemukan film yang mereka sukai dan akhirnya tidak melakukan pembelian atau langganan.

### Mengapa dan bagaimana mengatasi masalah tersebut:
Hal inilah yang mendorong munculnya sistem rekomendasi. Dimana, sistem rekomendasi memainkan peranan yang sangat penting saat ini. Sistem-sistem ini dirancang untuk menganalisis data pengguna dan film, serta memberikan saran film yang dipersonalisasi berdasarkan preferensi dan kebiasaan menonton. Namun, banyak sistem rekomendasi yang ada saat ini masih kurang akurat dan relevan, sehingga seringkali gagal memenuhi harapan pengguna.

Sebagai contoh saat ini, Netflix, Hulu, dan Amazon Prime Video dapat menggunakan AI untuk menganalisis riwayat menonton pengguna, rating film, dan interaksi mereka dengan platform untuk membuat rekomendasi yang lebih personal. Penerapan teknologi AI dan machine learning dapat membantu membangun sistem rekomendasi yang lebih personal, akurat, dan adaptif, sehingga memberikan pengalaman menonton film yang lebih optimal bagi pengguna.

## Business Understanding

### Problem Statements
Berdasarkan uraian yang telah dipaparkan pada latar belakang diatas, maka dapat diambil sebuah rumusan masalah yang dirumuskan sebagai berikut:

- Bagaimana cara model machine learning sistem rekomendasi dapat memberikan rekomendasi yang sesuai dengan selera pengguna?
- Bagaimana hasil evaluasi model machine learning dalam menawarkan rekomendasi film yang ingin pengguna tonton secara akurat dan relevan?

### Goals
Berdasarkan rumusan masalah yang telah dipaparkan di atas, maka proyek penelitian ini memiliki tujuan, yaitu:

- Mengetahui cara kerja model machine learning dalam membantu pengguna menemukan film yang tepat.
- Mengetahui hasil evaluasi metode pemfilteran berbasis konten dan kolaboratif dapat menawarkan rekomendasi film.

### Solution statements
Berdasarkan tujuan yang telah dipaparkan diatas, maka proyek penelitian ini memiliki solusi atau tahapan sebagai berikut:

1. Mengembangkan sistem rekomendasi dengan menganalisis data pengguna dan film secara lebih mendalam yang sesuai dengan selera pengguna.
   Algoritma yang akan dipakai diantaranya adalah sebagai berikut:
  - _Content-Based Filtering_
  - _Collaborative Filtering_

2. Evaluasi Model Sistem Rekomendasi dalam menampilkan hasil rekomendasi berdasarkan model yang telah dikembangkan.

## Data Understanding
Dataset yang dipakai dalam proyek machine learning ini merupakan dataset berjudul TMDB 5000 Movie Dataset. Dataset ini dipubilkasikan oleh THE MOVIE DATABASE (TMDB) melalui platform Kaggle. Dataset ini terdiri dari 2 file berformat csv (comma-separated values) yaitu `tmdb_5000_credits.csv` dan `tmdb_5000_movies.csv` dengan ukuran total 9 MB.

Link: [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata).

**_Tabel 1. Tampilan dataset `tmdb_5000_movies.csv` dalam bentuk dataframe dengan pandas_**

|   | budget  | genres	  | homepage	  | id  | keywords  | original_language  | original_title  | overview  | popularity  | production_companies  | production_countries  | release_date  | revenue  | runtime  | spoken_languages  | status  | tagline  | title  | vote_average  | vote_count  |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | 237000000	| [{"id": 28, "name": "Action"}, {"id": 12, "nam... | http://www.avatarmovie.com/	                | 19995	    | [{"id": 1463, "name": "culture clash"}, {"id":...	| en	| Avatar	                                | In the 22nd century, a paraplegic Marine is di...	| 150.437577	| [{"name": "Ingenious Film Partners", "id": 289...	| [{"iso_3166_1": "US", "name": "United States o...	| 2009-12-10	| 2787965087	| 162.0	| [{"iso_639_1": "en", "name": "English"}, {"iso...	| Released	| Enter the World of Pandora.	                    | Avatar	                                | 7.2	| 11800 |
| 1 | 300000000	| [{"id": 12, "name": "Adventure"}, {"id": 14, "... | http://disney.go.com/disneypictures/pirates/	| 285	    | [{"id": 270, "name": "ocean"}, {"id": 726, "na...	| en	| Pirates of the Caribbean: At World's End	| Captain Barbossa, long believed to be dead, ha...	| 139.082615	| [{"name": "Walt Disney Pictures", "id": 2}, {"...	| [{"iso_3166_1": "US", "name": "United States o...	| 2007-05-19	| 961000000	    | 169.0	| [{"iso_639_1": "en", "name": "English"}]	        | Released	| At the end of the world, the adventure begins.	| Pirates of the Caribbean: At World's End	| 6.9	| 4500  |
| 2 | 245000000	| [{"id": 28, "name": "Action"}, {"id": 12, "nam... | http://www.sonypictures.com/movies/spectre/	| 206647	| [{"id": 470, "name": "spy"}, {"id": 818, "name...	| en	| Spectre	                                | A cryptic message from Bond’s past sends him o...	| 107.376788	| [{"name": "Columbia Pictures", "id": 5}, {"nam...	| [{"iso_3166_1": "GB", "name": "United Kingdom"...	| 2015-10-26	| 880674609	    | 148.0	| [{"iso_639_1": "fr", "name": "Fran\u00e7ais"},...	| Released	| A Plan No One Escapes	                            | Spectre	                                | 6.3	| 4466  |

### Variabel-variabel pada dataset adalah sebagai berikut:
Pada file `tmdb_5000_movies.csv` berisi daftar film yang memiliki 4803 records data dan 20 feature, antara lain:
- `budget`: Numerical value representing the estimated production cost of the movie.
- `genres`: List of categories the movie belongs to (e.g., Action, Comedy, Drama).
- `homepage`: URL of the movie's official website (if available).
- `id`: Unique identifier for the movie within the TMDB database.
- `keywords`: List of relevant keywords associated with the movie.
- `original_language`: String indicating the language the movie was originally produced in.
- `original_title`: String representing the title the movie was originally released under.
- `overview`: Brief textual description of the movie's plot.
- `popularity`: Numerical score reflecting the movie's popularity on TMDB.
- `production_companies`: List of companies involved in producing the movie.
- `production_countries`: List of countries where the movie was produced.
- `release_date`: Date the movie was released in theaters.
- `revenue`: Numerical value representing the box office revenue generated by the movie.
- `runtime`: Numerical value indicating the movie's duration in minutes.
- `spoken_languages`: List of languages spoken within the movie.
- `status`: String indicating the current status of the movie (e.g., Released, Upcoming).
- `tagline`: Catchy phrase or slogan associated with the movie.
- `title`: String representing the movie's title as displayed on TMDB.
- `vote_average`: Numerical score representing the average rating given by users on TMDB.
- `vote_count`: Numerical value representing the total number of votes received by the movie on TMDB.

**_Tabel 2. Tampilan dataset `tmdb_5000_credits.csv` dalam bentuk dataframe dengan pandas_**

| # |  movie_id | title | cast | crew |
| --- | --- | --- | --- | --- |
| 0	| 19995	    | Avatar	                                | [{"cast_id": 242, "character": "Jake Sully", "...	| [{"credit_id": "52fe48009251416c750aca23", "de... |
| 1	| 285	    | Pirates of the Caribbean: At World's End	| [{"cast_id": 4, "character": "Captain Jack Spa...	| [{"credit_id": "52fe4232c3a36847f800b579", "de... |
| 2	| 206647	| Spectre	                                | [{"cast_id": 1, "character": "James Bond", "cr...	| [{"credit_id": "54805967c3a36829b5002c41", "de... |

### Variabel-variabel pada dataset adalah sebagai berikut:
Pada file `tmdb_5000_credits.csv` berisi daftar film yang memiliki 4803 records data dan 4 feature, antara lain:
- `movie_id`: Unique movie identifier for the movie within the TMDB database.
- `title`: String representing the movie's title as displayed on TMDB.
- `cast`: List of actors (names, characters, etc.).
- `crew`: List of behind-the-scenes personnel (names, job titles, etc.).

**_Tabel 3. Informasi dari dataset `tmdb_5000_movies.csv`_**

| # |  Column | Non-Null Count | Dtype  |
| --- | --- | --- | --- |
| 0   | budget                | 4803 non-null   | int64   |
| 1   | genres                | 4803 non-null   | object  |
| 2   | homepage              | 1712 non-null   | object  |
| 3   | id                    | 4803 non-null   | int64   |
| 4   | keywords              | 4803 non-null   | object  |
| 5   | original_language     | 4803 non-null   | object  |
| 6   | original_title        | 4803 non-null   | object  |
| 7   | overview              | 4800 non-null   | object  |
| 8   | popularity            | 4803 non-null   | float64 |
| 9   | production_companies  | 4803 non-null   | object  |
| 10  | production_countries  | 4803 non-null   | object  |
| 11  | release_date          | 4802 non-null   | object  |
| 12  | revenue               | 4803 non-null   | int64   |
| 13  | runtime               | 4801 non-null   | float64 |
| 14  | spoken_languages      | 4803 non-null   | object  |
| 15  | status                | 4803 non-null   | object  |
| 16  | tagline               | 3959 non-null   | object  |
| 17  | title                 | 4803 non-null   | object  |
| 18  | vote_average          | 4803 non-null   | float64 |
| 19  | vote_count            | 4803 non-null   | int64   |

Berdasarkan pada tabel di atas, dapat diketahui bahwa :
* Terdapat 13 kolom dengan tipe objek yaitu : genres, homepage, keywords, original_language, original_title, overview, production_companies, production_countries, release_date, spoken_languages, status, tagline, dan title. Kolom ini merupakan  _categorical features_ (fitur non-numerik).
* Terdapat 4 kolom bertipe numerik dengan tipe data int64 yaitu budget, id, revenue, dan vote_count. Kolom ini merupakan  _Numerical Features_ (fitur numerik).
* Terdapat 3 kolom bertipe numerik dengan tipe data float64 yaitu popularity, runtime, dan vote_average. Kolom ini merupakan  _Numerical Features_ (fitur numerik).

**_Tabel 4. Informasi dari dataset `tmdb_5000_credits.csv`_**

| # |  Column | Non-Null Count | Dtype  |
| --- | --- | --- | --- |
| 0   | movie_id  | 4803 non-null   | int64  |
| 1   | title     | 4803 non-null   | object |
| 2   | cast      | 4803 non-null   | object |
| 3   | crew      | 4803 non-null   | object |

 Berdasarkan pada tabel di atas, dapat diketahui bahwa :
* Terdapat 3 kolom dengan tipe objek yaitu : title, cast, dan crew. kolom ini merupakan  _categorical features_ (fitur non-numerik).
* Terdapat 1 kolom bertipe numerik dengan tipe data int64 yaitu movie_id. Kolom ini merupakan  _Numerical Features_ (fitur numerik).

### Visualisasi data atau Exploratory Data Analysis
#### Univariate Exploratory Data Analysis
**Melihat ada berapa banyak entri yang unik berdasarkan Banyak data dan List jenis unik**
```
budget 
--------------------
Banyak data budget: 436
List jenis budget: [237000000 300000000 245000000 250000000 260000000 258000000 280000000 
     31192     27000     22000     12000        13     20000      7000
    220000      9000]


genres 
--------------------
Banyak data genres: 1175
List jenis genres: ['[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]'
 ...
 '[{"id": 35, "name": "Comedy"}, {"id": 18, "name": "Drama"}, {"id": 10749, "name": "Romance"}, {"id": 10770, "name": "TV Movie"}]']


homepage 
--------------------
Banyak data homepage: 1692
List jenis homepage: ['http://www.avatarmovie.com/'
 'http://disney.go.com/disneypictures/pirates/'
 'http://www.sonypictures.com/movies/spectre/' ...
 'http://www.primermovie.com'
 'http://www.hallmarkchannel.com/signedsealeddelivered'
 'http://shanghaicalling.com/']


id 
--------------------
Banyak data id: 4803
List jenis id: [ 19995    285 206647 ... 231617 126186  25975]


keywords 
--------------------
Banyak data keywords: 4222
List jenis keywords: ['[{"id": 1463, "name": "culture clash"}, {"id": 2964, "name": "future"}, {"id": 3386, "name": "space war"}, {"id": 3388, "name": "space colony"}, {"id": 3679, "name": "society"}, {"id": 3801, "name": "space travel"}, {"id": 9685, "name": "futuristic"}, {"id": 9840, "name": "romance"}, {"id": 9882, "name": "space"}, {"id": 9951, "name": "alien"}, {"id": 10148, "name": "tribe"}, {"id": 10158, "name": "alien planet"}, {"id": 10987, "name": "cgi"}, {"id": 11399, "name": "marine"}, {"id": 13065, "name": "soldier"}, {"id": 14643, "name": "battle"}, {"id": 14720, "name": "love affair"}, {"id": 165431, "name": "anti war"}, {"id": 193554, "name": "power relations"}, {"id": 206690, "name": "mind and soul"}, {"id": 209714, "name": "3d"}]'
 ...
 '[{"id": 1523, "name": "obsession"}, {"id": 2249, "name": "camcorder"}, {"id": 9986, "name": "crush"}, {"id": 11223, "name": "dream girl"}]']


original_language 
--------------------
Banyak data original_language: 37
List jenis original_language: ['en' 'ja' 'fr' 'zh' 'es' 'de' 'hi' 'ru' 'ko' 'te' 'cn' 'it' 'nl' 'ta'
 'sv' 'th' 'da' 'xx' 'hu' 'cs' 'pt' 'is' 'tr' 'nb' 'af' 'pl' 'he' 'ar'
 'vi' 'ky' 'id' 'ro' 'fa' 'no' 'sl' 'ps' 'el']


original_title 
--------------------
Banyak data original_title: 4801
List jenis original_title: ['Avatar' "Pirates of the Caribbean: At World's End" 'Spectre' ...
 'Signed, Sealed, Delivered' 'Shanghai Calling' 'My Date with Drew']


overview 
--------------------
Banyak data overview: 4801
List jenis overview: ['In the 22nd century, a paraplegic Marine is dispatched to the moon Pandora on a unique mission, but becomes torn between following orders and protecting an alien civilization.' 
 ... 
 "Ever since the second grade when he first saw her in E.T. The Extraterrestrial, Brian Herzlinger has had a crush on Drew Barrymore. Now, 20 years later he's decided to try to fulfill his lifelong dream by asking her for a date. There's one small problem: She's Drew Barrymore and he's, well, Brian Herzlinger, a broke 27-year-old aspiring filmmaker from New Jersey."]


popularity 
--------------------
Banyak data popularity: 4802
List jenis popularity: [150.437577 139.082615 107.376788 ...   1.444476   0.857008   1.929883]


production_companies 
--------------------
Banyak data production_companies: 3697
List jenis production_companies: ['[{"name": "Ingenious Film Partners", "id": 289}, {"name": "Twentieth Century Fox Film Corporation", "id": 306}, {"name": "Dune Entertainment", "id": 444}, {"name": "Lightstorm Entertainment", "id": 574}]' 
 ... 
 '[{"name": "rusty bear entertainment", "id": 87986}, {"name": "lucky crow films", "id": 87987}]']


production_countries 
--------------------
Banyak data production_countries: 469
List jenis production_countries: ['[{"iso_3166_1": "US", "name": "United States of America"}, {"iso_3166_1": "GB", "name": "United Kingdom"}]' 
 '[]' 
 '[{"iso_3166_1": "US", "name": "United States of America"}, {"iso_3166_1": "CN", "name": "China"}]']


release_date 
--------------------
Banyak data release_date: 3281
List jenis release_date: ['2009-12-10' '2007-05-19' '2015-10-26' ... '2011-12-26' '2013-10-13'
 '2012-05-03']


revenue 
--------------------
Banyak data revenue: 3297
List jenis revenue: [2787965087  961000000  880674609 ...      99000     424760    2040920]


runtime 
--------------------
Banyak data runtime: 157
List jenis runtime: [162. 169. 148. 165. 132. 139. 100. 141. 153. 151. 154. 106. 149. 143. 
  47.  64.  60.]


spoken_languages 
--------------------
Banyak data spoken_languages: 544
List jenis spoken_languages: ['[{"iso_639_1": "en", "name": "English"}, {"iso_639_1": "es", "name": "Espa\\u00f1ol"}]' 
 '[]' 
 '[{"iso_639_1": "en", "name": "English"}, {"iso_639_1": "de", "name": "Deutsch"}, {"iso_639_1": "mn", "name": ""}]']


status 
--------------------
Banyak data status: 3
List jenis status: ['Released' 'Post Production' 'Rumored']


tagline 
--------------------
Banyak data tagline: 3945
List jenis tagline: ['Enter the World of Pandora.'
 ...
 'A New Yorker in Shanghai']


title 
--------------------
Banyak data title: 4800
List jenis title: ['Avatar' "Pirates of the Caribbean: At World's End" 'Spectre' ...
 'Signed, Sealed, Delivered' 'Shanghai Calling' 'My Date with Drew']


vote_average 
--------------------
Banyak data vote_average: 71
List jenis vote_average: [ 7.2  6.9  6.3  7.6  6.1  5.9  7.4  7.3  5.7  5.4  7.   6.5  6.4  6.2
  7.1  5.8  6.6  7.5  5.5  6.7  6.8  6.   5.1  7.8  5.6  5.2  8.2  7.7
  2.4]


vote_count 
--------------------
Banyak data vote_count: 1609
List jenis vote_count: [11800  4500  4466 ...   587  1708  2078]
```

Berdasarkan output movies dataframe di atas, diketahui beberapa informasi jumlah data unik, antara lain:

- Terdapat 436 jumlah budget.
- Terdapat 1175 jumlah genres.
- Terdapat 1692 jumlah homepage.
- Terdapat 4803 jumlah id.
- Terdapat 4222 jumlah keywords.
- Terdapat 37 jumlah original_language.
- Terdapat 4801 jumlah original_title.
- Terdapat 4801 jumlah overview.
- Terdapat 4802 jumlah popularity.
- Terdapat 3697 jumlah production_companies.
- Terdapat 469 jumlah production_countries.
- Terdapat 3281 jumlah release_date.
- Terdapat 3297 jumlah revenue.
- Terdapat 157 jumlah runtime.
- Terdapat 544 jumlah spoken_languages.
- Terdapat 3 jumlah status.
- Terdapat 3945 jumlah tagline.
- Terdapat 4800 jumlah title.
- Terdapat 71 jumlah vote_average.
- Terdapat 1609 jumlah vote_count.

```
movie_id 
--------------------
Banyak data movie_id: 4803
List jenis movie_id: [ 19995    285 206647 ... 231617 126186  25975]


title 
--------------------
Banyak data title: 4800
List jenis title: ['Avatar' "Pirates of the Caribbean: At World's End" 'Spectre' ...
 'Signed, Sealed, Delivered' 'Shanghai Calling' 'My Date with Drew']


cast 
--------------------
Banyak data cast: 4761
List jenis cast: ['[{"cast_id": 242, "character": "Jake Sully", "credit_id": "5602a8a7c3a3685532001c9a", "gender": 2, "id": 65731, "name": "Sam Worthington", "order": 0}, {"cast_id": 3, "character": "Neytiri", "credit_id": "52fe48009251416c750ac9cb", "gender": 1, "id": 8691, "name": "Zoe Saldana", "order": 1},
 ... 
"credit_id": "58ce0164c3a3685104015b28", "gender": 2, "id": 155007, "name": "Bill D\'Elia", "order": 7}]']


crew 
--------------------
Banyak data crew: 4776
List jenis crew: ['[{"credit_id": "52fe48009251416c750aca23", "department": "Editing", "gender": 0, "id": 1721, "job": "Editor", "name": "Stephen E. Rivkin"}' 
 ... 
{"credit_id": "52fe44e8c3a368484e03da97", "department": "Directing", "gender": 0, "id": 997560, "job": "Director", "name": "Brett Winn"}]']
```

Berdasarkan output credits dataframe di atas, diketahui beberapa informasi jumlah data unik, antara lain:

- Terdapat 4803 jumlah movie_id.
- Terdapat 4800 jumlah title.
- Terdapat 4761 jumlah cast.
- Terdapat 4776 jumlah crew.

**Melihat distribusi rating pada data, menggunakan fungsi describe()**

```
count    4803.000000
mean        6.092172
std         1.194612
min         0.000000
25%         5.600000
50%         6.200000
75%         6.800000
max        10.000000
```

Dari output di atas, diketahui bahwa nilai maksimum rating adalah 10 dan nilai minimumnya adalah 0. Artinya, skala rating berkisar antara 0 hingga 10.

## Data Preparation
Pada bagian ini penerapan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook ini adalah Content-Based Filtering dan Collaborative Filtering.

- **Data Merging**
Dua kumpulan data, yang mewakili movies dan credits yang berkaitan (pemain dan kru), digabungkan melalui operasi penggabungan berdasarkan atribut judul yang cocok. Proses ini akan menciptakan informasi dalam satu kumpulan data yang berisi metadata movies dan credits.

**_Tabel 5. Output Hasil Formating**
| | budget	| genres	| homepage	| id	| keywords	| original_language	| original_title	| overview	| popularity	| production_companies	| ...	| runtime	| spoken_languages	| status	| tagline	| title	| vote_average	| vote_count	| movie_id	| cast	| crew |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0	| 237000000	| [{"id": 28, "name": "Action"}, {"id": 12, "nam...	| http://www.avatarmovie.com/	                | 19995	    | [{"id": 1463, "name": "culture clash"}, {"id":...	| en	| Avatar	                                | In the 22nd century, a paraplegic Marine is di...	| 150.437577	| [{"name": "Ingenious Film Partners", "id": 289...	| ...	| 162.0	| [{"iso_639_1": "en", "name": "English"}, {"iso...	| Released	| Enter the World of Pandora.	                    | Avatar	                                | 7.2	| 11800	| 19995	    | [{"cast_id": 242, "character": "Jake Sully", "...	| [{"credit_id": "52fe48009251416c750aca23", "de... |
| 1	| 300000000	| [{"id": 12, "name": "Adventure"}, {"id": 14, "...	| http://disney.go.com/disneypictures/pirates/	| 285	    | [{"id": 270, "name": "ocean"}, {"id": 726, "na...	| en	| Pirates of the Caribbean: At World's End	| Captain Barbossa, long believed to be dead, ha...	| 139.082615	| [{"name": "Walt Disney Pictures", "id": 2}, {"...	| ...	| 169.0	| [{"iso_639_1": "en", "name": "English"}]	        | Released	| At the end of the world, the adventure begins.	| Pirates of the Caribbean: At World's End	| 6.9	| 4500	| 285	    | [{"cast_id": 4, "character": "Captain Jack Spa...	| [{"credit_id": "52fe4232c3a36847f800b579", "de... |
| 2	| 245000000	| [{"id": 28, "name": "Action"}, {"id": 12, "nam...	| http://www.sonypictures.com/movies/spectre/	| 206647	| [{"id": 470, "name": "spy"}, {"id": 818, "name...	| en	| Spectre	                                | A cryptic message from Bond’s past sends him o...	| 107.376788	| [{"name": "Columbia Pictures", "id": 5}, {"nam...	| ...	| 148.0	| [{"iso_639_1": "fr", "name": "Fran\u00e7ais"},...	| Released	| A Plan No One Escapes	                            | Spectre	                                | 6.3	| 4466	| 206647	| [{"cast_id": 1, "character": "James Bond", "cr...	| [{"credit_id": "54805967c3a36829b5002c41", "de... |

**Output Hasil Merging**

```
(4809, 23)
```

- **Data Selection**
Memilih hanya kolom yang akan diperlukan, seperti judul, genre, dan rating rata-rata. Proses ini akan meningkatkan efisiensi komputasi dan memfokuskan analisis pada variabel terkait.

**_Tabel 6. Output Hasil Selection_**
| | movie_id	| vote_average	| id	| title	| genres |
| --- | --- | --- | --- | --- | --- |
| 0 | 19995  | 7.2	   | 19995   | Avatar   | [{"id": 28, "name": "Action"}, {"id": 12, "nam...	|
| 1 | 285   | 6.9   | 285   | Pirates of the Caribbean: At World's End	   | [{"id": 12, "name": "Adventure"}, {"id": 14, "...	|
| 2 | 206647   | 6.3   | 206647   | Spectre   | [{"id": 28, "name": "Action"}, {"id": 12, "nam... |


- **Data Sorting**
Sort DataFrame berdasarkan `movie_id` .
Mengurutkan DataFrame berdasarkan `movie_id` secara ascending akan dapat dengan cepat menemukan film tertentu sehingga memudahkan navigasi dan referensinya. Proses ini akan memfasilitasi pengindeksan dan pengambilan data untuk analisis dan pemodelan selanjutnya.

**_Tabel 7. Output Hasil Sorting**
| | movie_id	| vote_average	| id	| title	| genres |
| --- | --- | --- | --- | --- | --- |
| 3773	| 5	| 6.5	| 5	| Four Rooms	| [{"id": 80, "name": "Crime"}, {"id": 35, "name... |
| 2917	| 11	| 8.1	| 11	| Star Wars	| [{"id": 12, "name": "Adventure"}, {"id": 28, "... |
| 328	| 12	| 7.6	| 12	| Finding Nemo	| [{"id": 16, "name": "Animation"}, {"id": 10751... |

- **Data Convertion**
Konversi kolom genres yang diperlukan menjadi list di mana setiap elemen genres akan dipisahkan dari elemen lain pada baris tersebut. Kemudian konversi kolom genres menjadi tipe string dari tipe list. Mengonversinya menjadi tipe string kemungkinan akan menggabungkan kategori-kategori ini menjadi satu string.

**_Tabel 8. Output Hasil Convertion**
| | movie_id	| vote_average	| id	| title	| genres |
| --- | --- | --- | --- | --- | --- |
| 3773	| 5	| 6.5	| 5	| Four Rooms	| Crime Comedy |
| 2917	| 11	| 8.1	| 11	| Star Wars	| Adventure Action ScienceFiction |
| 328	| 12	| 7.6	| 12	| Finding Nemo	| Animation Family |

- **Data Formating**
Membulatkan values pada kolom `vote_average` menjadi bilangan bulat. Misalnya, nilai 3,8 akan dibulatkan menjadi 4. Hal ini dapat berguna untuk visualisasi atau penyajian data dalam format yang lebih sederhana. 

**_Tabel 9. Output Hasil Formating**
| | movie_id	| vote_average	| id	| title	| genres |
| --- | --- | --- | --- | --- | --- |
| 3773	| 5	| 7.0	| 5	| Four Rooms	| Crime Comedy |
| 2917	| 11	| 9.0	| 11	| Star Wars	| Adventure Action ScienceFiction |
| 328	| 12	| 8.0	| 12	| Finding Nemo	| Animation Family |

- **Data Imputation**
Mengisi nilai yang hilang dari data yang tidak ada atau tidak lengkap. Proses ini akan mengganti nilai yang hilang dengan rata-rata, median, atau nilai lain yang sesuai berdasarkan konteks data.

**Output Missing value**

```
movie_id        0
vote_average    0
id              0
title           0
genres          0
```

Berdasarkan output di atas dapat dilihat bahwa tidak ditemukannya missing value pada masing masing kolom di dataset.

- **Data Cleaning**
Menghapus data duplikat yang identik dalam kumpulan data  Hal ini dapat terjadi karena kesalahan atau ketidakkonsistenan pada saat pengumpulan data. Menghapus duplikat memastikan analisis akurat dan menghindari hasil yang menyimpang.

### Perbedaan pada proses data preparation ada pada tahap akhir berikut :
**_Content Based Filtering_**

- **Data Convertion**
Menyiapkan informasi dalam format yang sesuai untuk pemrosesan dengan mengambil tiga kolom tertentu, 'movie_id', 'title', dan 'genres', dari DataFrame bernama 'new_df'. Kemudian mengubah setiap kolom menjadi format daftar terpisah menggunakan metode tolist(). Terakhir, menghitung dan mencetak panjang setiap daftar agar dapat memastikan konsistensi dalam jumlah entri di ketiga kategori.

**Output Data Convertion**

```
4809
4809
4809
```

**_Collaborative Filtering_**

- **Data Encoding**
Melakukan encoding pada 'user_ids' dan 'movie_id' yang berbeda menggunakan teknik Label Encoding dengan menetapkan bilangan bulat unik untuk setiap pengguna/film. Misalnya, pengguna dengan ID yang mirip mungkin memiliki selera yang sama, dan film dengan ID yang berurutan mungkin berasal dari genre yang sama.

Menggunakan Label Encoding dikarenakan, dibandingkan dengan one-hot encoding, yang membuat vektor biner terpisah untuk setiap ID unik, Label Encoding memerlukan lebih sedikit penyimpanan memori. Hal ini penting untuk kumpulan data besar dengan banyak ID.

**Output Data Encoding**

```
list userID:  [5, 11, 12, 13, 14,..., 426067, 426469, 433715, 447027, 459488]
encoded userID :  {5: 0, 11: 1, 12: 2, 13: 3, 14: 4,..., 426067: 4796, 426469: 4797, 433715: 4798, 447027: 4799, 459488: 4800}
encoded angka ke userID:  {0: 5, 1: 11, 2: 12, 3: 13, 4: 14,..., 4796: 426067, 4797: 426469, 4798: 433715, 4799: 447027, 4800: 459488}
```

## Modeling
Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan.

**Model Development dengan _Content Based Filtering_**

_Content Based Filtering_ (CBF) adalah salah satu teknik dalam Machine Learning yang digunakan untuk membangun model rekomendasi. Model CBF merekomendasikan item kepada pengguna berdasarkan kesamaan isi (content) item tersebut dengan item yang disukai pengguna sebelumnya.

**Kelebihan & Kekurangan _Content Based Filtering_**

**Kelebihan**
- CBF memiliki konsep dan logika yang intuitif sehingga mudah dipahami dan diterapkan.
- Rekomendasi CBF didasarkan pada kesamaan konten, sehingga mudah diinterpretasikan oleh pengguna dan dipahami alasan di balik rekomendasinya.
- CBF dapat bekerja dengan baik pada data yang sparse, yaitu data dengan jumlah item atau pengguna yang terbatas.
- CBF tidak memerlukan data eksplisit tentang preferensi pengguna, seperti rating atau ulasan, sehingga dapat digunakan pada situasi di mana data tersebut tidak tersedia.

**Kekurangan**
- CBF kesulitan merekomendasikan item kepada pengguna baru karena tidak memiliki informasi tentang preferensi mereka.
- Meskipun CBF efisien pada data sparse, performanya bisa menurun drastis pada data yang terlalu sparse.
- CBF dapat terlalu fokus pada item yang disukai pengguna sebelumnya, sehingga rekomendasi menjadi kurang variatif dan monoton.
- CBF hanya memperhitungkan konten item, sehingga tidak dapat mempertimbangkan faktor eksternal seperti tren populer atau preferensi sosial.

**Tahapan dalam Content Based Filtering (CBF):**

1. Pengumpulan data: Data tentang item yang akan direkomendasikan dikumpulkan. Data ini dapat berupa teks, gambar, video, atau jenis data lainnya. Data harus memiliki informasi tentang isi item, seperti genre film, topik artikel, atau deskripsi produk.
2. Preprocessing data: Data yang dikumpulkan dibersihkan dan diubah menjadi format yang dapat diproses oleh algoritma machine learning. Ini termasuk langkah-langkah seperti menghapus noise, normalisasi data, dan konversi data ke format numerik.


3. Pemilihan algoritma: Algoritma CBF yang sesuai dipilih berdasarkan jenis data dan tujuan rekomendasi. Algoritma populer termasuk TF-IDF, Cosine similarity, dan Jaccard similarity.

Algoritma yang digunakan pada kasus ini adalah **TF-IDF dan Cosine similarity**.

**_TF-IDF_** (Term Frequency-Inverse Document Frequency) adalah sebuah metode untuk menghitung bobot kata dalam sebuah dokumen. Bobot ini menunjukkan seberapa penting kata tersebut dalam mewakili isi dokumen.

**_Cosine Similarity_** adalah sebuah metode untuk mengukur kemiripan antara dua dokumen berdasarkan bobot kata-katanya.

**Alasan memilih TF-IDF dan Cosine Similarity:**

- Akurasi: TF-IDF dan Cosine Similarity umumnya memberikan hasil yang akurat dalam berbagai aplikasi NLP.
- Efisiensi: TF-IDF dan Cosine Similarity adalah metode yang efisien dan dapat digunakan untuk memproses data dalam jumlah besar.
- Interpretabilitas: Hasil TF-IDF dan Cosine Similarity mudah dipahami dan diinterpretasikan.

**Kelebihan TF-IDF dan Cosine Similarity:**

- Sederhana dan mudah diimplementasikan: TF-IDF dan Cosine Similarity adalah metode yang relatif sederhana dan mudah diimplementasikan.
- Efektif untuk data teks: TF-IDF dan Cosine Similarity telah terbukti efektif untuk berbagai aplikasi pemrosesan bahasa alami (NLP) yang melibatkan data teks.
- Fleksibel: TF-IDF dan Cosine Similarity dapat dikombinasikan dengan metode lain untuk meningkatkan kinerja.

**Kekurangan TF-IDF dan Cosine Similarity:**

- Sensitive terhadap noise: TF-IDF dan Cosine Similarity dapat sensitif terhadap noise dalam data teks.
- Sparse data: TF-IDF dan Cosine Similarity dapat bekerja kurang optimal pada data yang sparse.
- Dimensi data: TF-IDF dan Cosine Similarity dapat menghasilkan vektor dengan dimensi yang tinggi, yang dapat memperlambat proses komputasi.

4. Pelatihan model: Model CBF dilatih pada data yang telah diproses. Model ini mempelajari hubungan antara fitur dan preferensi pengguna.

Mengubah vektor tf-idf dalam bentuk matriks dengan fungsi todense()
```
matrix([[0.        , 0.        , 0.        , ..., 0.        , 0.        ,
         0.        ],
        [0.4954229 , 0.57326753, 0.        , ..., 0.        , 0.        ,
         0.        ],
        [0.        , 0.        , 0.7787053 , ..., 0.        , 0.        ,
         0.        ],
        ...,
        [0.        , 0.        , 0.        , ..., 0.        , 0.        ,
         0.        ],
        [0.        , 0.        , 0.        , ..., 0.        , 0.        ,
         0.        ],
        [0.        , 0.        , 0.        , ..., 0.        , 0.        ,
         0.        ]])
```

Menghitung cosine similarity pada matrix tf-idf

```
array([[1.        , 0.        , 0.        , ..., 0.        , 0.        ,
        0.        ],
       [0.        , 1.        , 0.        , ..., 0.        , 0.        ,
        0.        ],
       [0.        , 0.        , 1.        , ..., 0.        , 0.62738987,
        0.        ],
       ...,
       [0.        , 0.        , 0.        , ..., 1.        , 0.        ,
        0.        ],
       [0.        , 0.        , 0.62738987, ..., 0.        , 1.        ,
        0.        ],
       [0.        , 0.        , 0.        , ..., 0.        , 0.        ,
        1.        ]])
```

5. Evaluasi model: Model CBF dievaluasi untuk memastikan keakuratan dan performanya. Ini dapat dilakukan dengan menggunakan metrik seperti precision, recall, dan F1-score.
6. Penerapan model: Model CBF diterapkan pada sistem rekomendasi. Model ini digunakan untuk merekomendasikan item kepada pengguna berdasarkan kesamaan isi item dengan item yang disukai pengguna sebelumnya.

**_Tabel 10. Output Hasil System Rekomendasi**
| | title	| genres |
| --- | --- | --- |
| 0	| Transformers: Revenge of the Fallen	| ScienceFiction Action Adventure |
| 1	| John Carter	| Action Adventure ScienceFiction |
| 2	| Star Trek	| ScienceFiction Action Adventure |
| 3	| X-Men: First Class	| Action ScienceFiction Adventure |
| 4	| Star Trek Into Darkness	| Action Adventure ScienceFiction |
| 5	| Captain America: Civil War	| Adventure Action ScienceFiction |
| 6	| Independence Day: Resurgence	| Action Adventure ScienceFiction |
| 7	| Star Wars: Episode II - Attack of the Clones	| Adventure Action ScienceFiction |
| 8	| Journey 2: The Mysterious Island	| Adventure Action ScienceFiction |
| 9	| Iron Man 3	| Action Adventure ScienceFiction |

**Model Development dengan _Collaborative Filtering_**

Collaborative Filtering (CF) adalah salah satu teknik dalam Machine Learning yang digunakan untuk membangun model rekomendasi. Model CF merekomendasikan item kepada pengguna berdasarkan kesamaan perilaku pengguna lain.

**Kelebihan & Kekurangan _Collaborative Filtering_**

**Kelebihan**
- CF dapat menghasilkan rekomendasi yang sangat akurat dan relevan dengan preferensi pengguna berdasarkan perilaku pengguna lain yang serupa.
- CF mampu merekomendasikan item yang beragam dan tidak hanya terbatas pada item yang sudah disukai pengguna sebelumnya. Ini membantu pengguna menemukan item baru yang mungkin mereka sukai.
- CF dapat bekerja dengan baik pada data yang besar dan kompleks, karena hanya membutuhkan data tentang interaksi pengguna dan item.
- CF tidak selalu membutuhkan data eksplisit seperti rating atau ulasan. Bahkan, interaksi seperti pembelian, klik, atau penambahan ke wishlist juga bisa digunakan.

**Kekurangan**
- CF kesulitan merekomendasikan item kepada pengguna baru atau item baru karena belum ada data interaksi yang cukup untuk menentukan preferensi mereka.
- CF dapat berkinerja buruk pada data yang sparse, yaitu data dengan jumlah interaksi pengguna-item yang terbatas. Ini karena algoritma tidak memiliki cukup informasi untuk membuat prediksi akurat.
- Rekomendasi CF terkadang sulit dijelaskan kepada pengguna karena didasarkan pada perilaku anonim pengguna lain.
- CF dapat memperkuat bias yang ada dalam data, misalnya jika mayoritas pengguna memiliki preferensi serupa, rekomendasi yang dihasilkan mungkin tidak beragam.

**Tahapan dalam Collaborative Filtering (CF):**

1. Pengumpulan data: Data tentang interaksi pengguna dengan item dikumpulkan. Data ini dapat berupa rating, ulasan, pembelian, klik, atau interaksi lain.
2. Preprocessing data: Data dibersihkan dan diubah menjadi format yang dapat diproses oleh algoritma machine learning. Ini termasuk langkah-langkah seperti normalisasi data, pengolahan data yang hilang, dan konversi data ke format numerik.
3. Pemilihan algoritma: Algoritma CF yang sesuai dipilih berdasarkan jenis data, tujuan rekomendasi, dan preferensi developer. Algoritma populer termasuk User-based CF, Item-based CF, dan Matrix factorization.

Algoritma yang digunakan pada kasus ini adalah **User-based CF**.

**_User-based CF_** adalah sebuah teknik dalam sistem rekomendasi yang memprediksi preferensi pengguna berdasarkan kesamaan dengan pengguna lain. Sistem ini mengidentifikasi pengguna yang memiliki minat dan perilaku serupa dengan pengguna target, kemudian merekomendasikan item yang disukai oleh pengguna serupa tersebut.

**Alasan memilih User-based CF:**

- Lebih personal: User-based CF memberikan rekomendasi yang lebih personal karena mempertimbangkan kesamaan dengan pengguna lain.
- Lebih adaptif terhadap perubahan selera: User-based CF dapat beradaptasi dengan cepat terhadap perubahan selera pengguna karena hanya mempertimbangkan rating terbaru.
- Lebih mudah diimplementasikan: User-based CF lebih mudah diimplementasikan dibandingkan dengan Item-based CF dan Matrix Factorization.

**Kelebihan User-based CF:**

- Lebih mudah dipahami: Konsep User-based CF lebih mudah dipahami dibandingkan dengan algoritma CF lainnya.
- Lebih interpretabel: Hasil rekomendasi dari User-based CF lebih mudah diinterpretasikan karena didasarkan pada kesamaan dengan pengguna lain.
- Lebih efektif untuk data yang sparse: User-based CF dapat bekerja dengan baik pada data yang sparse, di mana terdapat banyak item yang belum dirating oleh pengguna.

**Kekurangan User-based CF:**

- Skalabilitas: User-based CF menjadi kurang skalabel ketika jumlah pengguna dan item dalam sistem meningkat.
- Sparse data: User-based CF dapat memberikan rekomendasi yang kurang akurat ketika data rating sangat sparse.
- Cold start problem: User-based CF sulit memberikan rekomendasi kepada pengguna baru yang belum memiliki banyak rating.

4. Pelatihan model: Model CF dilatih pada data yang telah diproses. Model ini mempelajari pola interaksi pengguna dan item untuk membuat prediksi tentang item mana yang mungkin disukai pengguna.

```
[[ 596  596]
 [3371 3372]
 [2701 2702]
 ...
 [3091 3092]
 [3770 3772]
 [ 859  860]] [0.7 0.7 0.8 ... 0.4 0.5 0.7]
```

5. Evaluasi model: Model CF dievaluasi untuk memastikan keakuratan dan performanya. Metrik yang umum digunakan termasuk Mean Absolute Error (MAE), Root Mean Square Error (RMSE), dan Precision & Recall.

![Metric Visualization](https://github.com/Wildanae123/Machine-Learning-Terapan-Predictive-Analytics/assets/104717412/8a9165ae-1932-480c-ba74-1342b5a6cd13)

**_Gambar 1. Metric Visualization_**

Berdasarkan visualisasi, pada proses training model didapatkan nilai error akhir sebesar sekitar 0.0315 dan error pada data validasi sebesar 0.1966

6. Penerapan model: Model CF diterapkan pada sistem rekomendasi. Model ini digunakan untuk merekomendasikan item kepada pengguna berdasarkan prediksi yang dihasilkan.

```
Showing recommendations for users: 133575
===========================
Movie with high ratings from user
--------------------------------
The Velocity of Gary : 5.0
--------------------------------
Top 10 movie recommendation
--------------------------------
The Godfather : Drama Crime
Schindler's List : Drama History War
The Good, the Bad and the Ugly : Western
The Green Mile : Fantasy Drama Crime
Fight Club : Drama
Modern Times : Drama Comedy
One Man's Hero : Western Action Drama History
Dancer, Texas Pop. 81 : Comedy Drama Family
Stiff Upper Lips : Comedy
Sardaarji :
```

## Evaluation

Evaluasi dilakukan untuk mengukur sejauh mana performance atau kinerja dari model sistem rekomendasi. Pada proyek ini, evaluasi diukur menggunakan metriks evaluasi sesuai dengan pendekatan yang dipakai dalam pengembangan sistem rekomendasi.

**Content-Based Filtering**

_Rumus:_
$$\text{Precision} = \frac{TP}{TP + FP}$$

_Keterangan:_

- _TP: True Positive (jumlah kasus positif yang diprediksi dengan benar)_
- _FP: False Positive (jumlah kasus positif yang salah prediksi)_

_Output:_
```
Precision: 0.3
```
_Rumus:_
$$\text{Recall} = \frac{TP}{TP + N}$$

_Keterangan:_

- _TP: True Positive (jumlah kasus positif yang diprediksi dengan benar)_
- _FP: False Positive (jumlah kasus positif yang salah prediksi)_
- _N : Jumlah kasus positif aktual (berapapun prediksinya)_

_Output:_
```
Recall: 0.08823529411764706
```
Rumus:
$$\text{F1-score} = 2 * \frac{Recall * Precision}{Recall + Precision}$$

_Output:_
```
F1-score: 0.13636363636363635
```

**Collaborative Filtering**

_Rumus:_
$$\text{MSE} = \frac{1}{N} \sum_{i=1}^N \left( y_i - \text{ypred}_i \right)^2$$

_Keterangan:_

- _N =  jumlah data_
- _y_i = nilai sebenarnya ke-i_
- _ypred_i = nilai prediksi ke-i_

_Output:_
```
Mean Absolute Error (MAE): 1.7290978749262307
```
_Rumus:_
$$\text{RMSE} = (\frac{\sum{}(y_i - \hat{y}_i)}{\text{n}})^{1/2}$$

_Keterangan:_

- _y  = nilai hasil observasi_
- _ŷ  = nilai hasil prediksi_
- _i  = urutan data pada database _
- _n  = jumlah data_

_Output:_
```
Root Mean Square Error (RMSE): 1.96600146116326
```

Berdasarkan hasil evaluasi di atas:

Content-Based Filtering memiliki precision yang relatif rendah (0.3), yang mengindikasikan bahwa dari item-item yang diprediksi relevan, hanya 30% yang benar-benar relevan. Recall-nya juga rendah (0.088), yang menunjukkan bahwa dari item-item yang seharusnya diprediksi relevan, hanya 8.8% yang berhasil diprediksi dengan benar. F1-score-nya (0.136) juga rendah, mencerminkan keseimbangan antara precision dan recall yang tidak optimal.

Collaborative Filtering memiliki Mean Absolute Error (MAE) sebesar 1.729 dan Root Mean Square Error (RMSE) sebesar 1.966. MAE dan RMSE yang rendah menunjukkan bahwa model Collaborative Filtering cenderung memiliki kesalahan prediksi yang lebih kecil dalam memperkirakan peringkat item oleh pengguna.

Berdasarkan hasil evaluasi tersebut, Collaborative Filtering lebih unggul dalam memprediksi preferensi pengguna karena memiliki tingkat kesalahan yang lebih rendah daripada Content-Based Filtering. Oleh karena itu, dalam konteks ini, Collaborative Filtering dapat dianggap sebagai metode yang lebih baik.

## Conclusion

Sistem rekomendasi film telah **berhasil** dikembangkan dengan menggunakan dua teknik Machine Learning, yaitu Content-Based Filtering (CBF) dan Collaborative Filtering (CF).

CBF merekomendasikan film berdasarkan kesamaan konten film dengan film yang disukai pengguna sebelumnya. Kelebihan CBF adalah mudah dipahami, bekerja dengan data sparse, dan tidak memerlukan data rating. Kekurangan CBF adalah kesulitan merekomendasikan film kepada pengguna baru, performanya menurun pada data sparse, dan rekomendasinya bisa monoton.

CF merekomendasikan film berdasarkan kesamaan perilaku pengguna lain. Kelebihan CF adalah menghasilkan rekomendasi yang akurat dan beragam, serta bekerja dengan baik pada data yang besar. Kekurangan CF adalah kesulitan merekomendasikan film kepada pengguna baru, berkinerja buruk pada data sparse, dan rekomendasinya sulit dijelaskan.

**Saran**
Namun, masih ada beberapa hal yang perlu diperbaiki. Sebagai saran kedepanya, sistem ini dapat dikembangkan lebih lanjut dengan menggunakan Metode Collaborative Filtering menggunakan rating film. Dengan menggunakan Metode Hibrid, pemfilteran berbasis konten dan kolaboratif dapat digunakan untuk sistem pemberi rekomendasi yang lebih efektif.

**---Ini adalah bagian akhir laporan---**

**Referensi:**

[1] [S. Katkam, A. Atikam, P. Mahesh, M. Chatre, S. S. Kumar and S. G. R, "Content-based Movie Recommendation System and Sentimental analysis using ML," 2023 7th International Conference on Intelligent Computing and Control Systems (ICICCS), Madurai, India, 2023, pp. 198-201.](https://doi.org/10.1109/ICICCS56967.2023.10142424)

[2] [simplilearn, "Netflix Recommendations: How Netflix Uses AI, Data Science, And ML," 7 November 2023. [Online]. Diakses pada 23 Februari 2024, dari https://www.simplilearn.com/how-netflix-uses-ai-data-science-and-ml-article](https://www.simplilearn.com/how-netflix-uses-ai-data-science-and-ml-article)

[3] [Agner, L., Necyk, B., Renzi, A. (2020). Recommendation Systems and Machine Learning: Mapping the User Experience. In: Marcus, A., Rosenzweig, E. (eds) Design, User Experience, and Usability. Design for Contemporary Interactive Environments. HCII 2020. Lecture Notes in Computer Science(), vol 12201. Springer, Cham.](https://doi.org/10.1007/978-3-030-49760-6_1)
