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
List jenis cast: ['[{"cast_id": 242, "character": "Jake Sully", "credit_id": "5602a8a7c3a3685532001c9a", "gender": 2, "id": 65731, "name": "Sam Worthington", "order": 0}, {"cast_id": 3, "character": "Neytiri", "credit_id": "52fe48009251416c750ac9cb", "gender": 1, "id": 8691, "name": "Zoe Saldana", "order": 1}, {"cast_id": 25, "character": "Dr. Grace Augustine", "credit_id": "52fe48009251416c750aca39", "gender": 1, "id": 10205, "name": "Sigourney Weaver", "order": 2}, {"cast_id": 4, "character": "Col. Quaritch", "credit_id": "52fe48009251416c750ac9cf", "gender": 2, "id": 32747, "name": "Stephen Lang", "order": 3}, {"cast_id": 5, "character": "Trudy Chacon", "credit_id": "52fe48009251416c750ac9d3", "gender": 1, "id": 17647, "name": "Michelle Rodriguez", "order": 4}, {"cast_id": 8, "character": "Selfridge", "credit_id": "52fe48009251416c750ac9e1", "gender": 2, "id": 1771, "name": "Giovanni Ribisi", "order": 5}, {"cast_id": 7, "character": "Norm Spellman", "credit_id": "52fe48009251416c750ac9dd", "gender": 2, "id": 59231, "name": "Joel David Moore", "order": 6}, {"cast_id": 9, "character": "Moat", "credit_id": "52fe48009251416c750ac9e5", "gender": 1, "id": 30485, "name": "CCH Pounder", "order": 7}, {"cast_id": 11, "character": "Eytukan", "credit_id": "52fe48009251416c750ac9ed", "gender": 2, "id": 15853, "name": "Wes Studi", "order": 8}, {"cast_id": 10, "character": "Tsu\'Tey", "credit_id": "52fe48009251416c750ac9e9", "gender": 2, "id": 10964, "name": "Laz Alonso", "order": 9}, {"cast_id": 12, "character": "Dr. Max Patel", "credit_id": "52fe48009251416c750ac9f1", "gender": 2, "id": 95697, "name": "Dileep Rao", "order": 10}, {"cast_id": 13, "character": "Lyle Wainfleet", "credit_id": "52fe48009251416c750ac9f5", "gender": 2, "id": 98215, "name": "Matt Gerald", "order": 11}, {"cast_id": 32, "character": "Private Fike", "credit_id": "52fe48009251416c750aca5b", "gender": 2, "id": 154153, "name": "Sean Anthony Moran", "order": 12}, {"cast_id": 33, "character": "Cryo Vault Med Tech", "credit_id": "52fe48009251416c750aca5f", "gender": 2, "id": 397312, "name": "Jason Whyte", "order": 13}, {"cast_id": 34, "character": "Venture Star Crew Chief", "credit_id": "52fe48009251416c750aca63", "gender": 2, "id": 42317, "name": "Scott Lawrence", "order": 14}, {"cast_id": 35, "character": "Lock Up Trooper", "credit_id": "52fe48009251416c750aca67", "gender": 2, "id": 986734, "name": "Kelly Kilgour", "order": 15}, {"cast_id": 36, "character": "Shuttle Pilot", "credit_id": "52fe48009251416c750aca6b", "gender": 0, "id": 1207227, "name": "James Patrick Pitt", "order": 16}, {"cast_id": 37, "character": "Shuttle Co-Pilot", "credit_id": "52fe48009251416c750aca6f", "gender": 0, "id": 1180936, "name": "Sean Patrick Murphy", "order": 17}, {"cast_id": 38, "character": "Shuttle Crew Chief", "credit_id": "52fe48009251416c750aca73", "gender": 2, "id": 1019578, "name": "Peter Dillon", "order": 18}, {"cast_id": 39, "character": "Tractor Operator / Troupe", "credit_id": "52fe48009251416c750aca77", "gender": 0, "id": 91443, "name": "Kevin Dorman", "order": 19}, {"cast_id": 40, "character": "Dragon Gunship Pilot", "credit_id": "52fe48009251416c750aca7b", "gender": 2, "id": 173391, "name": "Kelson Henderson", "order": 20}, {"cast_id": 41, "character": "Dragon Gunship Gunner", "credit_id": "52fe48009251416c750aca7f", "gender": 0, "id": 1207236, "name": "David Van Horn", "order": 21}, {"cast_id": 42, "character": "Dragon Gunship Navigator", "credit_id": "52fe48009251416c750aca83", "gender": 0, "id": 215913, "name": "Jacob Tomuri", "order": 22}, {"cast_id": 43, "character": "Suit #1", "credit_id": "52fe48009251416c750aca87", "gender": 0, "id": 143206, "name": "Michael Blain-Rozgay", "order": 23}, {"cast_id": 44, "character": "Suit #2", "credit_id": "52fe48009251416c750aca8b", "gender": 2, "id": 169676, "name": "Jon Curry", "order": 24}, {"cast_id": 46, "character": "Ambient Room Tech", "credit_id": "52fe48009251416c750aca8f", "gender": 0, "id": 1048610, "name": "Luke Hawker", "order": 25}, {"cast_id": 47, "character": "Ambient Room Tech / Troupe", "credit_id": "52fe48009251416c750aca93", "gender": 0, "id": 42288, "name": "Woody Schultz", "order": 26}, {"cast_id": 48, "character": "Horse Clan Leader", "credit_id": "52fe48009251416c750aca97", "gender": 2, "id": 68278, "name": "Peter Mensah", "order": 27}, {"cast_id": 49, "character": "Link Room Tech", "credit_id": "52fe48009251416c750aca9b", "gender": 0, "id": 1207247, "name": "Sonia Yee", "order": 28}, {"cast_id": 50, "character": "Basketball Avatar / Troupe", "credit_id": "52fe48009251416c750aca9f", "gender": 1, "id": 1207248, "name": "Jahnel Curfman", "order": 29}, {"cast_id": 51, "character": "Basketball Avatar", "credit_id": "52fe48009251416c750acaa3", "gender": 0, "id": 89714, "name": "Ilram Choi", "order": 30}, {"cast_id": 52, "character": "Na\'vi Child", "credit_id": "52fe48009251416c750acaa7", "gender": 0, "id": 1207249, "name": "Kyla Warren", "order": 31}, {"cast_id": 53, "character": "Troupe", "credit_id": "52fe48009251416c750acaab", "gender": 0, "id": 1207250, "name": "Lisa Roumain", "order": 32}, {"cast_id": 54, "character": "Troupe", "credit_id": "52fe48009251416c750acaaf", "gender": 1, "id": 83105, "name": "Debra Wilson", "order": 33}, {"cast_id": 57, "character": "Troupe", "credit_id": "52fe48009251416c750acabb", "gender": 0, "id": 1207253, "name": "Chris Mala", "order": 34}, {"cast_id": 55, "character": "Troupe", "credit_id": "52fe48009251416c750acab3", "gender": 0, "id": 1207251, "name": "Taylor Kibby", "order": 35}, {"cast_id": 56, "character": "Troupe", "credit_id": "52fe48009251416c750acab7", "gender": 0, "id": 1207252, "name": "Jodie Landau", "order": 36}, {"cast_id": 58, "character": "Troupe", "credit_id": "52fe48009251416c750acabf", "gender": 0, "id": 1207254, "name": "Julie Lamm", "order": 37}, {"cast_id": 59, "character": "Troupe", "credit_id": "52fe48009251416c750acac3", "gender": 0, "id": 1207257, "name": "Cullen B. Madden", "order": 38}, {"cast_id": 60, "character": "Troupe", "credit_id": "52fe48009251416c750acac7", "gender": 0, "id": 1207259, "name": "Joseph Brady Madden", "order": 39}, {"cast_id": 61, "character": "Troupe", "credit_id": "52fe48009251416c750acacb", "gender": 0, "id": 1207262, "name": "Frankie Torres", "order": 40}, {"cast_id": 62, "character": "Troupe", "credit_id": "52fe48009251416c750acacf", "gender": 1, "id": 1158600, "name": "Austin Wilson", "order": 41}, {"cast_id": 63, "character": "Troupe", "credit_id": "52fe48019251416c750acad3", "gender": 1, "id": 983705, "name": "Sara Wilson", "order": 42}, {"cast_id": 64, "character": "Troupe", "credit_id": "52fe48019251416c750acad7", "gender": 0, "id": 1207263, "name": "Tamica Washington-Miller", "order": 43}, {"cast_id": 65, "character": "Op Center Staff", "credit_id": "52fe48019251416c750acadb", "gender": 1, "id": 1145098, "name": "Lucy Briant", "order": 44}, {"cast_id": 66, "character": "Op Center Staff", "credit_id": "52fe48019251416c750acadf", "gender": 2, "id": 33305, "name": "Nathan Meister", "order": 45}, {"cast_id": 67, "character": "Op Center Staff", "credit_id": "52fe48019251416c750acae3", "gender": 0, "id": 1207264, "name": "Gerry Blair", "order": 46}, {"cast_id": 68, "character": "Op Center Staff", "credit_id": "52fe48019251416c750acae7", "gender": 2, "id": 33311, "name": "Matthew Chamberlain", "order": 47}, {"cast_id": 69, "character": "Op Center Staff", "credit_id": "52fe48019251416c750acaeb", "gender": 0, "id": 1207265, "name": "Paul Yates", "order": 48}, {"cast_id": 70, "character": "Op Center Duty Officer", "credit_id": "52fe48019251416c750acaef", "gender": 0, "id": 1207266, "name": "Wray Wilson", "order": 49}, {"cast_id": 71, "character": "Op Center Staff", "credit_id": "52fe48019251416c750acaf3", "gender": 2, "id": 54492, "name": "James Gaylyn", "order": 50}, {"cast_id": 72, "character": "Dancer", "credit_id": "52fe48019251416c750acaf7", "gender": 0, "id": 1207267, "name": "Melvin Leno Clark III", "order": 51}, {"cast_id": 73, "character": "Dancer", "credit_id": "52fe48019251416c750acafb", "gender": 0, "id": 1207268, "name": "Carvon Futrell", "order": 52}, {"cast_id": 74, "character": "Dancer", "credit_id": "52fe48019251416c750acaff", "gender": 0, "id": 1207269, "name": "Brandon Jelkes", "order": 53}, {"cast_id": 75, "character": "Dancer", "credit_id": "52fe48019251416c750acb03", "gender": 0, "id": 1207270, "name": "Micah Moch", "order": 54}, {"cast_id": 76, "character": "Dancer", "credit_id": "52fe48019251416c750acb07", "gender": 0, "id": 1207271, "name": "Hanniyah Muhammad", "order": 55}, {"cast_id": 77, "character": "Dancer", "credit_id": "52fe48019251416c750acb0b", "gender": 0, "id": 1207272, "name": "Christopher Nolen", "order": 56}, {"cast_id": 78, "character": "Dancer", "credit_id": "52fe48019251416c750acb0f", "gender": 0, "id": 1207273, "name": "Christa Oliver", "order": 57}, {"cast_id": 79, "character": "Dancer", "credit_id": "52fe48019251416c750acb13", "gender": 0, "id": 1207274, "name": "April Marie Thomas", "order": 58}, {"cast_id": 80, "character": "Dancer", "credit_id": "52fe48019251416c750acb17", "gender": 0, "id": 1207275, "name": "Bravita A. Threatt", "order": 59}, {"cast_id": 81, "character": "Mining Chief (uncredited)", "credit_id": "52fe48019251416c750acb1b", "gender": 0, "id": 1207276, "name": "Colin Bleasdale", "order": 60}, {"cast_id": 82, "character": "Veteran Miner (uncredited)", "credit_id": "52fe48019251416c750acb1f", "gender": 0, "id": 107969, "name": "Mike Bodnar", "order": 61}, {"cast_id": 83, "character": "Richard (uncredited)", "credit_id": "52fe48019251416c750acb23", "gender": 0, "id": 1207278, "name": "Matt Clayton", "order": 62}, {"cast_id": 84, "character": "Nav\'i (uncredited)", "credit_id": "52fe48019251416c750acb27", "gender": 1, "id": 147898, "name": "Nicole Dionne", "order": 63}, {"cast_id": 85, "character": "Trooper (uncredited)", "credit_id": "52fe48019251416c750acb2b", "gender": 0, "id": 1207280, "name": "Jamie Harrison", "order": 64}, {"cast_id": 86, "character": "Trooper (uncredited)", "credit_id": "52fe48019251416c750acb2f", "gender": 0, "id": 1207281, "name": "Allan Henry", "order": 65}, {"cast_id": 87, "character": "Ground Technician (uncredited)", "credit_id": "52fe48019251416c750acb33", "gender": 2, "id": 1207282, "name": "Anthony Ingruber", "order": 66}, {"cast_id": 88, "character": "Flight Crew Mechanic (uncredited)", "credit_id": "52fe48019251416c750acb37", "gender": 0, "id": 1207283, "name": "Ashley Jeffery", "order": 67}, {"cast_id": 14, "character": "Samson Pilot", "credit_id": "52fe48009251416c750ac9f9", "gender": 0, "id": 98216, "name": "Dean Knowsley", "order": 68}, {"cast_id": 89, "character": "Trooper (uncredited)", "credit_id": "52fe48019251416c750acb3b", "gender": 0, "id": 1201399, "name": "Joseph Mika-Hunt", "order": 69}, {"cast_id": 90, "character": "Banshee (uncredited)", "credit_id": "52fe48019251416c750acb3f", "gender": 0, "id": 236696, "name": "Terry Notary", "order": 70}, {"cast_id": 91, "character": "Soldier (uncredited)", "credit_id": "52fe48019251416c750acb43", "gender": 0, "id": 1207287, "name": "Kai Pantano", "order": 71}, {"cast_id": 92, "character": "Blast Technician (uncredited)", "credit_id": "52fe48019251416c750acb47", "gender": 0, "id": 1207288, "name": "Logan Pithyou", "order": 72}, {"cast_id": 93, "character": "Vindum Raah (uncredited)", "credit_id": "52fe48019251416c750acb4b", "gender": 0, "id": 1207289, "name": "Stuart Pollock", "order": 73}, {"cast_id": 94, "character": "Hero (uncredited)", "credit_id": "52fe48019251416c750acb4f", "gender": 0, "id": 584868, "name": "Raja", "order": 74}, {"cast_id": 95, "character": "Ops Centreworker (uncredited)", "credit_id": "52fe48019251416c750acb53", "gender": 0, "id": 1207290, "name": "Gareth Ruck", "order": 75}, {"cast_id": 96, "character": "Engineer (uncredited)", "credit_id": "52fe48019251416c750acb57", "gender": 0, "id": 1062463, "name": "Rhian Sheehan", "order": 76}, {"cast_id": 97, "character": "Col. Quaritch\'s Mech Suit (uncredited)", "credit_id": "52fe48019251416c750acb5b", "gender": 0, "id": 60656, "name": "T. J. Storm", "order": 77}, {"cast_id": 98, "character": "Female Marine (uncredited)", "credit_id": "52fe48019251416c750acb5f", "gender": 0, "id": 1207291, "name": "Jodie Taylor", "order": 78}, {"cast_id": 99, "character": "Ikran Clan Leader (uncredited)", "credit_id": "52fe48019251416c750acb63", "gender": 1, "id": 1186027, "name": "Alicia Vela-Bailey", "order": 79}, {"cast_id": 100, "character": "Geologist (uncredited)", "credit_id": "52fe48019251416c750acb67", "gender": 0, "id": 1207292, "name": "Richard Whiteside", "order": 80}, {"cast_id": 101, "character": "Na\'vi (uncredited)", "credit_id": "52fe48019251416c750acb6b", "gender": 0, "id": 103259, "name": "Nikie Zambo", "order": 81}, {"cast_id": 102, "character": "Ambient Room Tech / Troupe", "credit_id": "52fe48019251416c750acb6f", "gender": 1, "id": 42286, "name": "Julene Renee", "order": 82}]' 
 ... 
 '[{"cast_id": 3, "character": "Herself", "credit_id": "52fe44e8c3a368484e03da91", "gender": 1, "id": 69597, "name": "Drew Barrymore", "order": 0}, {"cast_id": 5, "character": "Himself", "credit_id": "58ce01169251415a3901648f", "gender": 2, "id": 85563, "name": "Brian Herzlinger", "order": 1}, {"cast_id": 6, "character": "Himself", "credit_id": "58ce01339251415a410167f0", "gender": 2, "id": 3034, "name": "Corey Feldman", "order": 2}, {"cast_id": 8, "character": "Himself", "credit_id": "58ce018c9251415a7d016e36", "gender": 2, "id": 21315, "name": "Eric Roberts", "order": 3}, {"cast_id": 9, "character": "Himself", "credit_id": "58ce01b99251415a7d016e7d", "gender": 0, "id": 2171, "name": "Griffin Dunne", "order": 4}, {"cast_id": 10, "character": "Himself", "credit_id": "58ce01d19251415a8b0168be", "gender": 2, "id": 2231, "name": "Samuel L. Jackson", "order": 5}, {"cast_id": 11, "character": "Himself", "credit_id": "58ce01dd9251415a39016580", "gender": 2, "id": 14407, "name": "Matt LeBlanc", "order": 6}, {"cast_id": 7, "character": "Himself", "credit_id": "58ce0164c3a3685104015b28", "gender": 2, "id": 155007, "name": "Bill D\'Elia", "order": 7}]']


crew 
--------------------
Banyak data crew: 4776
List jenis crew: ['[{"credit_id": "52fe48009251416c750aca23", "department": "Editing", "gender": 0, "id": 1721, "job": "Editor", "name": "Stephen E. Rivkin"}, {"credit_id": "539c47ecc3a36810e3001f87", "department": "Art", "gender": 2, "id": 496, "job": "Production Design", "name": "Rick Carter"}, {"credit_id": "54491c89c3a3680fb4001cf7", "department": "Sound", "gender": 0, "id": 900, "job": "Sound Designer", "name": "Christopher Boyes"}, {"credit_id": "54491cb70e0a267480001bd0", "department": "Sound", "gender": 0, "id": 900, "job": "Supervising Sound Editor", "name": "Christopher Boyes"}, {"credit_id": "539c4a4cc3a36810c9002101", "department": "Production", "gender": 1, "id": 1262, "job": "Casting", "name": "Mali Finn"}, {"credit_id": "5544ee3b925141499f0008fc", "department": "Sound", "gender": 2, "id": 1729, "job": "Original Music Composer", "name": "James Horner"}, {"credit_id": "52fe48009251416c750ac9c3", "department": "Directing", "gender": 2, "id": 2710, "job": "Director", "name": "James Cameron"}, {"credit_id": "52fe48009251416c750ac9d9", "department": "Writing", "gender": 2, "id": 2710, "job": "Writer", "name": "James Cameron"}, {"credit_id": "52fe48009251416c750aca17", "department": "Editing", "gender": 2, "id": 2710, "job": "Editor", "name": "James Cameron"}, {"credit_id": "52fe48009251416c750aca29", "department": "Production", "gender": 2, "id": 2710, "job": "Producer", "name": "James Cameron"}, {"credit_id": "52fe48009251416c750aca3f", "department": "Writing", "gender": 2, "id": 2710, "job": "Screenplay", "name": "James Cameron"}, {"credit_id": "539c4987c3a36810ba0021a4", "department": "Art", "gender": 2, "id": 7236, "job": "Art Direction", "name": "Andrew Menzies"}, {"credit_id": "549598c3c3a3686ae9004383", "department": "Visual Effects", "gender": 0, "id": 6690, "job": "Visual Effects Producer", "name": "Jill Brooks"}, {"credit_id": "52fe48009251416c750aca4b", "department": "Production", "gender": 1, "id": 6347, "job": "Casting", "name": "Margery Simkin"}, {"credit_id": "570b6f419251417da70032fe", "department": "Art", "gender": 2, "id": 6878, "job": "Supervising Art Director", "name": "Kevin Ishioka"}, {"credit_id": "5495a0fac3a3686ae9004468", "department": "Sound", "gender": 0, "id": 6883, "job": "Music Editor", "name": "Dick Bernstein"}, {"credit_id": "54959706c3a3686af3003e81", "department": "Sound", "gender": 0, "id": 8159, "job": "Sound Effects Editor", "name": "Shannon Mills"}, {"credit_id": "54491d58c3a3680fb1001ccb", "department": "Sound", "gender": 0, "id": 8160, "job": "Foley", "name": "Dennie Thorpe"}, {"credit_id": "54491d6cc3a3680fa5001b2c", "department": "Sound", "gender": 0, "id": 8163, "job": "Foley", "name": "Jana Vance"}, {"credit_id": "52fe48009251416c750aca57", "department": "Costume & Make-Up", "gender": 1, "id": 8527, "job": "Costume Design", "name": "Deborah Lynn Scott"}, {"credit_id": "52fe48009251416c750aca2f", "department": "Production", "gender": 2, "id": 8529, "job": "Producer", "name": "Jon Landau"}, {"credit_id": "539c4937c3a36810ba002194", "department": "Art", "gender": 0, "id": 9618, "job": "Art Direction", "name": "Sean Haworth"}, {"credit_id": "539c49b6c3a36810c10020e6", "department": "Art", "gender": 1, "id": 12653, "job": "Set Decoration", "name": "Kim Sinclair"}, {"credit_id": "570b6f2f9251413a0e00020d", "department": "Art", "gender": 1, "id": 12653, "job": "Supervising Art Director", "name": "Kim Sinclair"}, {"credit_id": "54491a6c0e0a26748c001b19", "department": "Art", "gender": 2, "id": 14350, "job": "Set Designer", "name": "Richard F. Mays"}, {"credit_id": "56928cf4c3a3684cff0025c4", "department": "Production", "gender": 1, "id": 20294, "job": "Executive Producer", "name": "Laeta Kalogridis"}, {"credit_id": "52fe48009251416c750aca51", "department": "Costume & Make-Up", "gender": 0, "id": 17675, "job": "Costume Design", "name": "Mayes C. Rubeo"}, {"credit_id": "52fe48009251416c750aca11", "department": "Camera", "gender": 2, "id": 18265, "job": "Director of Photography", "name": "Mauro Fiore"}, {"credit_id": "5449194d0e0a26748f001b39", "department": "Art", "gender": 0, "id": 42281, "job": "Set Designer", "name": "Scott Herbertson"}, {"credit_id": "52fe48009251416c750aca05", "department": "Crew", "gender": 0, "id": 42288, "job": "Stunts", "name": "Woody Schultz"}, {"credit_id": "5592aefb92514152de0010f5", "department": "Costume & Make-Up", "gender": 0, "id": 29067, "job": "Makeup Artist", "name": "Linda DeVetta"}, {"credit_id": "5592afa492514152de00112c", "department": "Costume & Make-Up", "gender": 0, "id": 29067, "job": "Hairstylist", "name": "Linda DeVetta"}, {"credit_id": "54959ed592514130fc002e5d", "department": "Camera", "gender": 2, "id": 33302, "job": "Camera Operator", "name": "Richard Bluck"}, {"credit_id": "539c4891c3a36810ba002147", "department": "Art", "gender": 2, "id": 33303, "job": "Art Direction", "name": "Simon Bright"}, {"credit_id": "54959c069251417a81001f3a", "department": "Visual Effects", "gender": 0, "id": 113145, "job": "Visual Effects Supervisor", "name": "Richard Martin"}, {"credit_id": "54959a0dc3a3680ff5002c8d", "department": "Crew", "gender": 2, "id": 58188, "job": "Visual Effects Editor", "name": "Steve R. Moore"}, {"credit_id": "52fe48009251416c750aca1d", "department": "Editing", "gender": 2, "id": 58871, "job": "Editor", "name": "John Refoua"}, {"credit_id": "54491a4dc3a3680fc30018ca", "department": "Art", "gender": 0, "id": 92359, "job": "Set Designer", "name": "Karl J. Martin"}, {"credit_id": "52fe48009251416c750aca35", "department": "Camera", "gender": 1, "id": 72201, "job": "Director of Photography", "name": "Chiling Lin"}, {"credit_id": "52fe48009251416c750ac9ff", "department": "Crew", "gender": 0, "id": 89714, "job": "Stunts", "name": "Ilram Choi"}, {"credit_id": "54959c529251416e2b004394", "department": "Visual Effects", "gender": 2, "id": 93214, "job": "Visual Effects Supervisor", "name": "Steven Quale"}, {"credit_id": "54491edf0e0a267489001c37", "department": "Crew", "gender": 1, "id": 122607, "job": "Dialect Coach", "name": "Carla Meyer"}, {"credit_id": "539c485bc3a368653d001a3a", "department": "Art", "gender": 2, "id": 132585, "job": "Art Direction", "name": "Nick Bassett"}, {"credit_id": "539c4903c3a368653d001a74", "department": "Art", "gender": 0, "id": 132596, "job": "Art Direction", "name": "Jill Cormack"}, {"credit_id": "539c4967c3a368653d001a94", "department": "Art", "gender": 0, "id": 132604, "job": "Art Direction", "name": "Andy McLaren"}, {"credit_id": "52fe48009251416c750aca45", "department": "Crew", "gender": 0, "id": 236696, "job": "Motion Capture Artist", "name": "Terry Notary"}, {"credit_id": "54959e02c3a3680fc60027d2", "department": "Crew", "gender": 2, "id": 956198, "job": "Stunt Coordinator", "name": "Garrett Warren"}, {"credit_id": "54959ca3c3a3686ae300438c", "department": "Visual Effects", "gender": 2, "id": 957874, "job": "Visual Effects Supervisor", "name": "Jonathan Rothbart"}, {"credit_id": "570b6f519251412c74001b2f", "department": "Art", "gender": 0, "id": 957889, "job": "Supervising Art Director", "name": "Stefan Dechant"}, {"credit_id": "570b6f62c3a3680b77007460", "department": "Art", "gender": 2, "id": 959555, "job": "Supervising Art Director", "name": "Todd Cherniawsky"}, {"credit_id": "539c4a3ac3a36810da0021cc", "department": "Production", "gender": 0, "id": 1016177, "job": "Casting", "name": "Miranda Rivers"}, {"credit_id": "539c482cc3a36810c1002062", "department": "Art", "gender": 0, "id": 1032536, "job": "Production Design", "name": "Robert Stromberg"}, {"credit_id": "539c4b65c3a36810c9002125", "department": "Costume & Make-Up", "gender": 2, "id": 1071680, "job": "Costume Design", "name": "John Harding"}, {"credit_id": "54959e6692514130fc002e4e", "department": "Camera", "gender": 0, "id": 1177364, "job": "Steadicam Operator", "name": "Roberto De Angelis"}, {"credit_id": "539c49f1c3a368653d001aac", "department": "Costume & Make-Up", "gender": 2, "id": 1202850, "job": "Makeup Department Head", "name": "Mike Smithson"}, {"credit_id": "5495999ec3a3686ae100460c", "department": "Visual Effects", "gender": 0, "id": 1204668, "job": "Visual Effects Producer", "name": "Alain Lalanne"}, {"credit_id": "54959cdfc3a3681153002729", "department": "Visual Effects", "gender": 0, "id": 1206410, "job": "Visual Effects Supervisor", "name": "Lucas Salton"}, {"credit_id": "549596239251417a81001eae", "department": "Crew", "gender": 0, "id": 1234266, "job": "Post Production Supervisor", "name": "Janace Tashjian"}, {"credit_id": "54959c859251416e1e003efe", "department": "Visual Effects", "gender": 0, "id": 1271932, "job": "Visual Effects Supervisor", "name": "Stephen Rosenbaum"}, {"credit_id": "5592af28c3a368775a00105f", "department": "Costume & Make-Up", "gender": 0, "id": 1310064, "job": "Makeup Artist", "name": "Frankie Karena"}, {"credit_id": "539c4adfc3a36810e300203b", "department": "Costume & Make-Up", "gender": 1, "id": 1319844, "job": "Costume Supervisor", "name": "Lisa Lovaas"}, {"credit_id": "54959b579251416e2b004371", "department": "Visual Effects", "gender": 0, "id": 1327028, "job": "Visual Effects Supervisor", "name": "Jonathan Fawkner"}, {"credit_id": "539c48a7c3a36810b5001fa7", "department": "Art", "gender": 0, "id": 1330561, "job": "Art Direction", "name": "Robert Bavin"}, {"credit_id": "539c4a71c3a36810da0021e0", "department": "Costume & Make-Up", "gender": 0, "id": 1330567, "job": "Costume Supervisor", "name": "Anthony Almaraz"}, {"credit_id": "539c4a8ac3a36810ba0021e4", "department": "Costume & Make-Up", "gender": 0, "id": 1330570, "job": "Costume Supervisor", "name": "Carolyn M. Fenton"}, {"credit_id": "539c4ab6c3a36810da0021f0", "department": "Costume & Make-Up", "gender": 0, "id": 1330574, "job": "Costume Supervisor", "name": "Beth Koenigsberg"}, {"credit_id": "54491ab70e0a267480001ba2", "department": "Art", "gender": 0, "id": 1336191, "job": "Set Designer", "name": "Sam Page"}, {"credit_id": "544919d9c3a3680fc30018bd", "department": "Art", "gender": 0, "id": 1339441, "job": "Set Designer", "name": "Tex Kadonaga"}, {"credit_id": "54491cf50e0a267483001b0c", "department": "Editing", "gender": 0, "id": 1352422, "job": "Dialogue Editor", "name": "Kim Foscato"}, {"credit_id": "544919f40e0a26748c001b09", "department": "Art", "gender": 0, "id": 1352962, "job": "Set Designer", "name": "Tammy S. Lee"}, {"credit_id": "5495a115c3a3680ff5002d71", "department": "Crew", "gender": 0, "id": 1357070, "job": "Transportation Coordinator", "name": "Denny Caira"}, {"credit_id": "5495a12f92514130fc002e94", "department": "Crew", "gender": 0, "id": 1357071, "job": "Transportation Coordinator", "name": "James Waitkus"}, {"credit_id": "5495976fc3a36811530026b0", "department": "Sound", "gender": 0, "id": 1360103, "job": "Supervising Sound Editor", "name": "Addison Teague"}, {"credit_id": "54491837c3a3680fb1001c5a", "department": "Art", "gender": 2, "id": 1376887, "job": "Set Designer", "name": "C. Scott Baker"}, {"credit_id": "54491878c3a3680fb4001c9d", "department": "Art", "gender": 0, "id": 1376888, "job": "Set Designer", "name": "Luke Caska"}, {"credit_id": "544918dac3a3680fa5001ae0", "department": "Art", "gender": 0, "id": 1376889, "job": "Set Designer", "name": "David Chow"}, {"credit_id": "544919110e0a267486001b68", "department": "Art", "gender": 0, "id": 1376890, "job": "Set Designer", "name": "Jonathan Dyer"}, {"credit_id": "54491967c3a3680faa001b5e", "department": "Art", "gender": 0, "id": 1376891, "job": "Set Designer", "name": "Joseph Hiura"}, {"credit_id": "54491997c3a3680fb1001c8a", "department": "Art", "gender": 0, "id": 1376892, "job": "Art Department Coordinator", "name": "Rebecca Jellie"}, {"credit_id": "544919ba0e0a26748f001b42", "department": "Art", "gender": 0, "id": 1376893, "job": "Set Designer", "name": "Robert Andrew Johnson"}, {"credit_id": "54491b1dc3a3680faa001b8c", "department": "Art", "gender": 0, "id": 1376895, "job": "Assistant Art Director", "name": "Mike Stassi"}, {"credit_id": "54491b79c3a3680fbb001826", "department": "Art", "gender": 0, "id": 1376897, "job": "Construction Coordinator", "name": "John Villarino"}, {"credit_id": "54491baec3a3680fb4001ce6", "department": "Art", "gender": 2, "id": 1376898, "job": "Assistant Art Director", "name": "Jeffrey Wisniewski"}, {"credit_id": "54491d2fc3a3680fb4001d07", "department": "Editing", "gender": 0, "id": 1376899, "job": "Dialogue Editor", "name": "Cheryl Nardi"}, {"credit_id": "54491d86c3a3680fa5001b2f", "department": "Editing", "gender": 0, "id": 1376901, "job": "Dialogue Editor", "name": "Marshall Winn"}, {"credit_id": "54491d9dc3a3680faa001bb0", "department": "Sound", "gender": 0, "id": 1376902, "job": "Supervising Sound Editor", "name": "Gwendolyn Yates Whittle"}, {"credit_id": "54491dc10e0a267486001bce", "department": "Sound", "gender": 0, "id": 1376903, "job": "Sound Re-Recording Mixer", "name": "William Stein"}, {"credit_id": "54491f500e0a26747c001c07", "department": "Crew", "gender": 0, "id": 1376909, "job": "Choreographer", "name": "Lula Washington"}, {"credit_id": "549599239251412c4e002a2e", "department": "Visual Effects", "gender": 0, "id": 1391692, "job": "Visual Effects Producer", "name": "Chris Del Conte"}, {"credit_id": "54959d54c3a36831b8001d9a", "department": "Visual Effects", "gender": 2, "id": 1391695, "job": "Visual Effects Supervisor", "name": "R. Christopher White"}, {"credit_id": "54959bdf9251412c4e002a66", "department": "Visual Effects", "gender": 0, "id": 1394070, "job": "Visual Effects Supervisor", "name": "Dan Lemmon"}, {"credit_id": "5495971d92514132ed002922", "department": "Sound", "gender": 0, "id": 1394129, "job": "Sound Effects Editor", "name": "Tim Nielsen"}, {"credit_id": "5592b25792514152cc0011aa", "department": "Crew", "gender": 0, "id": 1394286, "job": "CG Supervisor", "name": "Michael Mulholland"}, {"credit_id": "54959a329251416e2b004355", "department": "Crew", "gender": 0, "id": 1394750, "job": "Visual Effects Editor", "name": "Thomas Nittmann"}, {"credit_id": "54959d6dc3a3686ae9004401", "department": "Visual Effects", "gender": 0, "id": 1394755, "job": "Visual Effects Supervisor", "name": "Edson Williams"}, {"credit_id": "5495a08fc3a3686ae300441c", "department": "Editing", "gender": 0, "id": 1394953, "job": "Digital Intermediate", "name": "Christine Carr"}, {"credit_id": "55402d659251413d6d000249", "department": "Visual Effects", "gender": 0, "id": 1395269, "job": "Visual Effects Supervisor", "name": "John Bruno"}, {"credit_id": "54959e7b9251416e1e003f3e", "department": "Camera", "gender": 0, "id": 1398970, "job": "Steadicam Operator", "name": "David Emmerichs"}, {"credit_id": "54959734c3a3686ae10045e0", "department": "Sound", "gender": 0, "id": 1400906, "job": "Sound Effects Editor", "name": "Christopher Scarabosio"}, {"credit_id": "549595dd92514130fc002d79", "department": "Production", "gender": 0, "id": 1401784, "job": "Production Supervisor", "name": "Jennifer Teves"}, {"credit_id": "549596009251413af70028cc", "department": "Production", "gender": 0, "id": 1401785, "job": "Production Manager", "name": "Brigitte Yorke"}, {"credit_id": "549596e892514130fc002d99", "department": "Sound", "gender": 0, "id": 1401786, "job": "Sound Effects Editor", "name": "Ken Fischer"}, {"credit_id": "549598229251412c4e002a1c", "department": "Crew", "gender": 0, "id": 1401787, "job": "Special Effects Coordinator", "name": "Iain Hutton"}, {"credit_id": "549598349251416e2b00432b", "department": "Crew", "gender": 0, "id": 1401788, "job": "Special Effects Coordinator", "name": "Steve Ingram"}, {"credit_id": "54959905c3a3686ae3004324", "department": "Visual Effects", "gender": 0, "id": 1401789, "job": "Visual Effects Producer", "name": "Joyce Cox"}, {"credit_id": "5495994b92514132ed002951", "department": "Visual Effects", "gender": 0, "id": 1401790, "job": "Visual Effects Producer", "name": "Jenny Foster"}, {"credit_id": "549599cbc3a3686ae1004613", "department": "Crew", "gender": 0, "id": 1401791, "job": "Visual Effects Editor", "name": "Christopher Marino"}, {"credit_id": "549599f2c3a3686ae100461e", "department": "Crew", "gender": 0, "id": 1401792, "job": "Visual Effects Editor", "name": "Jim Milton"}, {"credit_id": "54959a51c3a3686af3003eb5", "department": "Visual Effects", "gender": 0, "id": 1401793, "job": "Visual Effects Producer", "name": "Cyndi Ochs"}, {"credit_id": "54959a7cc3a36811530026f4", "department": "Crew", "gender": 0, "id": 1401794, "job": "Visual Effects Editor", "name": "Lucas Putnam"}, {"credit_id": "54959b91c3a3680ff5002cb4", "department": "Visual Effects", "gender": 0, "id": 1401795, "job": "Visual Effects Supervisor", "name": "Anthony \'Max\' Ivins"}, {"credit_id": "54959bb69251412c4e002a5f", "department": "Visual Effects", "gender": 0, "id": 1401796, "job": "Visual Effects Supervisor", "name": "John Knoll"}, {"credit_id": "54959cbbc3a3686ae3004391", "department": "Visual Effects", "gender": 2, "id": 1401799, "job": "Visual Effects Supervisor", "name": "Eric Saindon"}, {"credit_id": "54959d06c3a3686ae90043f6", "department": "Visual Effects", "gender": 0, "id": 1401800, "job": "Visual Effects Supervisor", "name": "Wayne Stables"}, {"credit_id": "54959d259251416e1e003f11", "department": "Visual Effects", "gender": 0, "id": 1401801, "job": "Visual Effects Supervisor", "name": "David Stinnett"}, {"credit_id": "54959db49251413af7002975", "department": "Visual Effects", "gender": 0, "id": 1401803, "job": "Visual Effects Supervisor", "name": "Guy Williams"}, {"credit_id": "54959de4c3a3681153002750", "department": "Crew", "gender": 0, "id": 1401804, "job": "Stunt Coordinator", "name": "Stuart Thorp"}, {"credit_id": "54959ef2c3a3680fc60027f2", "department": "Lighting", "gender": 0, "id": 1401805, "job": "Best Boy Electric", "name": "Giles Coburn"}, {"credit_id": "54959f07c3a3680fc60027f9", "department": "Camera", "gender": 2, "id": 1401806, "job": "Still Photographer", "name": "Mark Fellman"}, {"credit_id": "54959f47c3a3681153002774", "department": "Lighting", "gender": 0, "id": 1401807, "job": "Lighting Technician", "name": "Scott Sprague"}, {"credit_id": "54959f8cc3a36831b8001df2", "department": "Visual Effects", "gender": 0, "id": 1401808, "job": "Animation Director", "name": "Jeremy Hollobon"}, {"credit_id": "54959fa0c3a36831b8001dfb", "department": "Visual Effects", "gender": 0, "id": 1401809, "job": "Animation Director", "name": "Orlando Meunier"}, {"credit_id": "54959fb6c3a3686af3003f54", "department": "Visual Effects", "gender": 0, "id": 1401810, "job": "Animation Director", "name": "Taisuke Tanimura"}, {"credit_id": "54959fd2c3a36831b8001e02", "department": "Costume & Make-Up", "gender": 0, "id": 1401812, "job": "Set Costumer", "name": "Lilia Mishel Acevedo"}, {"credit_id": "54959ff9c3a3686ae300440c", "department": "Costume & Make-Up", "gender": 0, "id": 1401814, "job": "Set Costumer", "name": "Alejandro M. Hernandez"}, {"credit_id": "5495a0ddc3a3686ae10046fe", "department": "Editing", "gender": 0, "id": 1401815, "job": "Digital Intermediate", "name": "Marvin Hall"}, {"credit_id": "5495a1f7c3a3686ae3004443", "department": "Production", "gender": 0, "id": 1401816, "job": "Publicist", "name": "Judy Alley"}, {"credit_id": "5592b29fc3a36869d100002f", "department": "Crew", "gender": 0, "id": 1418381, "job": "CG Supervisor", "name": "Mike Perry"}, {"credit_id": "5592b23a9251415df8001081", "department": "Crew", "gender": 0, "id": 1426854, "job": "CG Supervisor", "name": "Andrew Morley"}, {"credit_id": "55491e1192514104c40002d8", "department": "Art", "gender": 0, "id": 1438901, "job": "Conceptual Design", "name": "Seth Engstrom"}, {"credit_id": "5525d5809251417276002b06", "department": "Crew", "gender": 0, "id": 1447362, "job": "Visual Effects Art Director", "name": "Eric Oliver"}, {"credit_id": "554427ca925141586500312a", "department": "Visual Effects", "gender": 0, "id": 1447503, "job": "Modeling", "name": "Matsune Suzuki"}, {"credit_id": "551906889251415aab001c88", "department": "Art", "gender": 0, "id": 1447524, "job": "Art Department Manager", "name": "Paul Tobin"}, {"credit_id": "5592af8492514152cc0010de", "department": "Costume & Make-Up", "gender": 0, "id": 1452643, "job": "Hairstylist", "name": "Roxane Griffin"}, {"credit_id": "553d3c109251415852001318", "department": "Lighting", "gender": 0, "id": 1453938, "job": "Lighting Artist", "name": "Arun Ram-Mohan"}, {"credit_id": "5592af4692514152d5001355", "department": "Costume & Make-Up", "gender": 0, "id": 1457305, "job": "Makeup Artist", "name": "Georgia Lockhart-Adams"}, {"credit_id": "5592b2eac3a36877470012a5", "department": "Crew", "gender": 0, "id": 1466035, "job": "CG Supervisor", "name": "Thrain Shadbolt"}, {"credit_id": "5592b032c3a36877450015f1", "department": "Crew", "gender": 0, "id": 1483220, "job": "CG Supervisor", "name": "Brad Alexander"}, {"credit_id": "5592b05592514152d80012f6", "department": "Crew", "gender": 0, "id": 1483221, "job": "CG Supervisor", "name": "Shadi Almassizadeh"}, {"credit_id": "5592b090c3a36877570010b5", "department": "Crew", "gender": 0, "id": 1483222, "job": "CG Supervisor", "name": "Simon Clutterbuck"}, {"credit_id": "5592b0dbc3a368774b00112c", "department": "Crew", "gender": 0, "id": 1483223, "job": "CG Supervisor", "name": "Graeme Demmocks"}, {"credit_id": "5592b0fe92514152db0010c1", "department": "Crew", "gender": 0, "id": 1483224, "job": "CG Supervisor", "name": "Adrian Fernandes"}, {"credit_id": "5592b11f9251415df8001059", "department": "Crew", "gender": 0, "id": 1483225, "job": "CG Supervisor", "name": "Mitch Gates"}, {"credit_id": "5592b15dc3a3687745001645", "department": "Crew", "gender": 0, "id": 1483226, "job": "CG Supervisor", "name": "Jerry Kung"}, {"credit_id": "5592b18e925141645a0004ae", "department": "Crew", "gender": 0, "id": 1483227, "job": "CG Supervisor", "name": "Andy Lomas"}, {"credit_id": "5592b1bfc3a368775d0010e7", "department": "Crew", "gender": 0, "id": 1483228, "job": "CG Supervisor", "name": "Sebastian Marino"}, {"credit_id": "5592b2049251415df8001078", "department": "Crew", "gender": 0, "id": 1483229, "job": "CG Supervisor", "name": "Matthias Menz"}, {"credit_id": "5592b27b92514152d800136a", "department": "Crew", "gender": 0, "id": 1483230, "job": "CG Supervisor", "name": "Sergei Nevshupov"}, {"credit_id": "5592b2c3c3a36869e800003c", "department": "Crew", "gender": 0, "id": 1483231, "job": "CG Supervisor", "name": "Philippe Rebours"}, {"credit_id": "5592b317c3a36877470012af", "department": "Crew", "gender": 0, "id": 1483232, "job": "CG Supervisor", "name": "Michael Takarangi"}, {"credit_id": "5592b345c3a36877470012bb", "department": "Crew", "gender": 0, "id": 1483233, "job": "CG Supervisor", "name": "David Weitzberg"}, {"credit_id": "5592b37cc3a368775100113b", "department": "Crew", "gender": 0, "id": 1483234, "job": "CG Supervisor", "name": "Ben White"}, {"credit_id": "573c8e2f9251413f5d000094", "department": "Crew", "gender": 1, "id": 1621932, "job": "Stunts", "name": "Min Windle"}]' 
 ... 
 '[{"credit_id": "58ce021b9251415a390165d9", "department": "Production", "gender": 2, "id": 6888, "job": "Executive Producer", "name": "Clark Peterson"}, {"credit_id": "58ce0232c3a36850e90157da", "department": "Production", "gender": 2, "id": 61051, "job": "Executive Producer", "name": "Andrew Reimer"}, {"credit_id": "52fe44e8c3a368484e03da8d", "department": "Directing", "gender": 2, "id": 85563, "job": "Director", "name": "Brian Herzlinger"}, {"credit_id": "52fe44e8c3a368484e03da87", "department": "Directing", "gender": 2, "id": 94471, "job": "Director", "name": "Jon Gunn"}, {"credit_id": "52fe44e8c3a368484e03da97", "department": "Directing", "gender": 0, "id": 997560, "job": "Director", "name": "Brett Winn"}]']
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
Pada bagian ini penerapan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

- **Data Merging**
Dua kumpulan data, yang mewakili movies dan credits yang berkaitan (pemain dan kru), digabungkan melalui operasi penggabungan berdasarkan atribut judul yang cocok. Proses ini akan menciptakan informasi dalam satu kumpulan data yang berisi metadata movies dan credits.

- **Data Selection**
Memilih hanya kolom yang akan diperlukan, seperti judul, genre, dan rating rata-rata. Proses ini akan meningkatkan efisiensi komputasi dan memfokuskan analisis pada variabel terkait.

- **Data Sorting**
Sort DataFrame berdasarkan `movie_id` secara ascending.
Jika Anda memiliki banyak film, mengurutkannya berdasarkan ID akan memudahkan navigasi dan referensinya. Anggap saja seperti mengurutkan perpustakaan berdasarkan abjad - Anda dapat dengan cepat menemukan film tertentu yang Anda cari.

Kumpulan data yang digabungkan diurutkan dalam urutan menaik berdasarkan atribut "movie_id".
Ini memfasilitasi pengindeksan dan pengambilan data untuk analisis dan pemodelan selanjutnya.

- **Data Convertion**
Konversi kolom yang diperlukan menjadi daftar string. Kemudian konversi kolom genres menjadi tipe string dari tipe list.

- **Data Formating**
Bulatkan values pada kolom `vote_average` menjadi bilangan bulat.

- **Data Cleaning**
Membersihkan data dengan mengisi missing values dan menghapus duplikat data.

## Modeling
Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan.

> [!NOTE]
> Pada proyek ini, algoritma machine learning yang dipakai adalah _K-Nearest Neighbor_, _Random Forest_, _Boosting Algorithm_ dan _Support Vector Machine_.
> Alasan Pemilihan Model:
> - _KNN_ dapat digunakan untuk menyelesaikan masalah klasifikasi dan regresi, sesuai dengan kebutuhan analisis.
> - _RF_ merupakan model yang memiliki akurasi dan stabilitasnya dalam menangani berbagai masalah klasifikasi dan regresi.
> - _AdaBoost_ dapat menghasilkan model dengan akurasi yang sangat tinggi.
> - _SVM_ menggunakan memori yang lebih sedikit dibandingkan model lain seperti _Random Forest_.

**Model Development dengan _K-Nearest Neighbor_**
_KNN_ bekerja dengan membandingkan jarak satu sampel ke sampel pelatihan lain dengan memilih sejumlah k tetangga terdekat (dengan k adalah sebuah angka positif). Nah, itulah mengapa algoritma ini dinamakan _K-nearest neighbor_ (sejumlah k tetangga terdekat). _KNN_ bisa digunakan untuk kasus klasifikasi dan regresi.

**Kelebihan & Kekurangan _KNN_**

`Kelebihan`
- Mudah dipahami dan diimplementasikan.
- Berkinerja baik pada data dengan dimensi rendah.
- Fleksibel dan dapat digunakan untuk klasifikasi dan regresi.

`Kekurangan`
- Sensitif terhadap outlier dan noise.
- Kinerja menurun pada data dengan dimensi tinggi.
- Membutuhkan waktu komputasi yang lama untuk data yang besar.

`Tahapan dan Parameter:`

1. Praproses data: Normalisasi data untuk memastikan semua fitur memiliki skala yang sama.
2. Pemilihan nilai K: Menentukan nilai K yang optimal melalui proses tuning. Nilai K yang dipilih adalah 5.
3. Klasifikasi data: Menerapkan algoritma _KNN_ dengan nilai K=5 untuk mengklasifikasikan data baru.

**Model Development dengan _Random Forest_**
Algoritma _Random forest_ adalah salah satu algoritma supervised learning. Ia dapat digunakan untuk menyelesaikan masalah klasifikasi dan regresi. _Random forest_ juga merupakan algoritma yang sering digunakan karena cukup sederhana tetapi memiliki stabilitas yang mumpuni.

**Kelebihan & Kekurangan _RF_**

`Kelebihan`
- Akurat dan robust.
- Mampu menangani data dengan dimensi tinggi.
- Dapat mengatasi overfitting.

`Kekurangan`
- Interpretasi model yang kompleks.
- Membutuhkan waktu training yang lama.
- Sensitif terhadap hyperparameter tuning.

`Tahapan dan Parameter:`

1. Praproses data: Normalisasi data untuk memastikan semua fitur memiliki skala yang sama.
2. Tuning hyperparameter: Menentukan nilai hyperparameter optimal seperti jumlah pohon (n_estimators) dan kedalaman pohon (max_depth) melalui proses tuning.
3. Klasifikasi data: Menerapkan algoritma _Random Forest_ dengan hyperparameter yang optimal untuk mengklasifikasikan data baru.

**Model Development dengan _Boosting Algorithm_**
Algoritma yang menggunakan teknik _Boosting_ bekerja dengan membangun model dari data latih. Kemudian ia membuat model kedua yang bertugas memperbaiki kesalahan dari model pertama. Model ditambahkan sampai data latih terprediksi dengan baik atau telah mencapai jumlah maksimum model untuk ditambahkan.

**Kelebihan & Kekurangan _AdaBOOST_**

`Kelebihan`
- Akurat dan dapat menangani data yang kompleks.
- Mampu mengatasi overfitting.
- Berkinerja baik pada data dengan noise.

`Kekurangan`
- Interpretasi model yang kompleks.
- Sensitif terhadap outlier.
- Membutuhkan waktu training yang lama.

`Tahapan dan Parameter:`

1. Praproses data: Normalisasi data untuk memastikan semua fitur memiliki skala yang sama.
2. Tuning hyperparameter: Menentukan nilai hyperparameter optimal seperti jumlah iterasi (n_estimators) dan learning rate melalui proses tuning.
3. Klasifikasi data: Menerapkan algoritma _Boosting_ seperti _AdaBoost_ dengan hyperparameter yang optimal untuk mengklasifikasikan data baru.

**Model Development dengan _Support Vector Machine_ (SVM)**
Tujuan dari algoritma SVM adalah untuk menemukan hyperplane terbaik dalam ruang berdimensi-N (ruang dengan N-jumlah fitur) yang berfungsi sebagai pemisah yang jelas bagi titik-titik data input.

**Kelebihan & Kekurangan _SVM_**

`Kelebihan`
- Akurat dan robust.
- Mampu menangani data dengan dimensi tinggi.
- Efisien dalam penggunaan memori.
 
`Kekurangan`
- Interpretasi model yang kompleks.
Membutuhkan waktu training yang lama untuk data yang besar.
Sensitif terhadap outlier dan noise.

`Tahapan dan Parameter:`

1. Praproses data: Normalisasi data dan scaling data untuk memastikan semua fitur memiliki skala yang sama.
2. Pemilihan kernel: Menentukan kernel yang tepat seperti linear kernel atau Gaussian kernel.
3. Tuning hyperparameter: Menentukan nilai hyperparameter optimal seperti regularization parameter (C) dan gamma melalui proses tuning.
4. Klasifikasi data: Menerapkan algoritma SVM dengan hyperparameter yang optimal untuk mengklasifikasikan data baru.

**Pemilihan Model::**
Pemilihan parameter yang optimal untuk setiap algoritma sangat penting untuk mencapai performa terbaik. Dari keempat model yang telah dilatih, prediksi dengan _K-Nearest Neighbor_ memberikan hasil yang paling mendekati. Sehingga **_K-Nearest Neighbor_** merupakan model terbaik yang dihasilkan.

## Evaluation
Sebelum model diterapkan, model perlu dievaluasi agar terbukti cocok untuk tujuan yang telah ditentukan. Fase ini bertujuan untuk memastikan bahwa model akan mampu membuat prediksi yang akurat dan tidak mengalami overfitting atau underfitting.

Mengevaluasi model regresi sebenarnya relatif sederhana. Secara umum, hampir semua metrik adalah sama. Jika prediksi mendekati nilai sebenarnya, performanya baik. Sedangkan jika tidak, performanya buruk. Secara teknis, selisih antara nilai sebenarnya dan nilai prediksi disebut eror. Maka, semua metrik mengukur seberapa kecil nilai eror tersebut.

Metrik yang akan akan digunakan pada prediksi ini adalah MSE atau Mean Squared Error yang menghitung jumlah selisih kuadrat rata-rata nilai sebenarnya dengan nilai prediksi. Mean Squared Error (MSE) adalah metrik yang digunakan untuk mengukur ketepatan model regresi dalam memprediksi nilai target. MSE menghitung rata-rata dari kuadrat perbedaan antara nilai prediksi dan nilai target yang sebenarnya.

Nilai MSE memiliki beberapa signifikansi, antara lain:
- Akurasi Prediksi: Nilai MSE yang rendah menunjukkan bahwa model mampu memprediksi nilai target dengan lebih akurat.
- Kecocokan Model: Nilai MSE yang rendah menunjukkan bahwa model cocok dengan data yang digunakan.
- Perbandingan Model: Nilai MSE dapat digunakan untuk membandingkan kinerja model yang berbeda.

MSE didefinisikan dalam persamaan berikut

$$\text{MSE} = \frac{1}{N} \sum_{i=1}^N \left( y_i - \text{ypred}_i \right)^2$$

_Keterangan:_

_N = jumlah dataset_

_yi = nilai sebenarnya_

_y_pred = nilai prediksi_

Hasil evaluasi pada data latih dan data test adalah sebagai berikut.

**_Tabel 6. Hasil Perhitungan MSE 4 Algoritma_**
| | train | test |
| ------------- | ------------- | ------------- |
| KNN | 4.751774 | 6.166513 |
| RF | 0.013235 | 0.081261 |
| Boosting | 3.795002 | 4.343941 |
| SVM | 49.703125 | 46.693444 |

Untuk memudahkan, gunakan plot matrik tersebut dengan bar chart

![Visualisasi bar chart MSE](https://github.com/Wildanae123/Machine-Learning-Terapan-Predictive-Analytics/assets/104717412/f84f41a2-f667-4cac-980d-153771b5fc7b)

**_Gambar 7. Visualisasi bar chart MSE_**

Dari gambar di atas , terlihat bahwa, model _Random Forest (RF)_ memiliki nilai error pada data test yang paling kecil sedangkan model _Support Vector Machine_ memiliki nilai error paling banyak dibandingkan dari ketiga model. Hal ini menunjukkan bahwa RF mampu memprediksi nilai target dengan lebih akurat dibandingkan dengan model lain.

Untuk mengujinya, buat prediksi menggunakan beberapa harga dari data test.

**_Tabel 7. Hasil Prediksi MSE_**
| x | y_true	 | prediksi_KNN | prediksi_RF | prediksi_Boosting | prediksi_SVM |
| --- | --- | --- | --- | --- | --- |
| 131 | 580.419 | 581.1 | 582.2 | 573.0 | 303.2 |

Pada Tabel di atas adalah hasil prediksi "Total" dari 4 algoritma yaitu _K-Nearest Neighbor_, _Random Forest_, _AdaBOOST_ dan _Support Vector Machine_. Terlihat bahwa prediksi dengan _Random Forest_ dan _K-Nearest Neighbor_ memberikan hasil yang paling mendekati. Dimana algoritma _K-Nearest Neighbor_ memiliki nilai prediksi _MSE_ (Mean Squared Error) sebesar 581.1, algoritma _Random Forest_ memiliki nilai prediksi _MSE_ (Mean Squared Error) sebesar 582.2, algoritma _AdaBOOST_ memiliki nilai prediksi _MSE_ (Mean Squared Error) sebesar 573.0 sedangkan algoritma _Support Vector Machine_ memiliki nilai prediksi _MSE_ (Mean Squared Error) sebesar 303.2.

**Kesimpulan**

Berdasarkan hasil evaluasi, proyek ini **berhasil** dalam mencapai tujuannya untuk mengembangkan model machine learning yang dapat membantu para peritel dalam optimasi proses bisnis dan layanan untuk meningkatkan kepuasan pelanggan. Dari keempat model Algoritma yang dikembangkan berdasarkan hasil perbandingan dan visualisasi, dapat disimpulkan bahwa model _Random Forest_ merupakan model terbaik dengan nilai error terkecil pada data test. Meskipun model _K-Nearest Neighbor_ memberikan akurasi yang mendekati, namun RF lebih unggul dalam hal akurasi yang tinggi dan nilai error yang rendah.

**Saran**
Namun, masih ada beberapa area yang perlu diperbaiki untuk meningkatkan performa model dan efektivitasnya.

- Melakukan tuning hyperparameter untuk meningkatkan performa model.
- Mencoba model regresi lainnya untuk dibandingkan dengan model yang telah diuji.

**---Ini adalah bagian akhir laporan---**

**Referensi:**

[1] [van Raaij, E.M. (2005), "The strategic value of customer profitability analysis", Marketing Intelligence & Planning, Vol. 23 No. 4, pp. 372-381.](https://doi.org/10.1108/02634500510603474)

[2] [K. Vergidis, A. Tiwari and B. Majeed, "Business Process Analysis and Optimization: Beyond Reengineering," in IEEE Transactions on Systems, Man, and Cybernetics, Part C (Applications and Reviews), vol. 38, no. 1, pp. 69-82, Jan. 2008.](https://doi.org/10.1109/TSMCC.2007.905812)

[3] [Krumeich, J., Werth, D. & Loos, P. Prescriptive Control of Business Processes. Bus Inf Syst Eng 58, 261–280 (2016).](https://doi.org/10.1007/s12599-015-0412-2)
