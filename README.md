# Proyek Akhir: Menyelesaikan Permasalahan Perusahaan Edutech

## Business Understanding
Jaya Jaya Institut adalah sebuah institusi pendidikan tinggi terkemuka yang telah beroperasi sejak tahun 2000. Selama lebih dari dua dekade perjalanannya, institusi ini telah membuktikan kualitasnya dengan mencetak banyak lulusan yang dikenal memiliki reputasi sangat baik di berbagai bidang. Meskipun demikian, di balik catatan keberhasilan tersebut, Jaya Jaya Institut juga menghadapi tantangan yang signifikan terkait retensi siswa. Terdapat kenyataan bahwa banyak siswa yang memulai pendidikan mereka tidak berhasil menyelesaikannya, atau mengalami fenomena dropout. Isu ini menjadi perhatian krusial yang perlu dianalisis lebih dalam.

### Permasalahan Bisnis
Tingginya angka dropout merupakan tantangan signifikan bagi institusi pendidikan seperti Jaya Jaya Institut, berpotensi memengaruhi reputasi dan efektivitas program studi. Menyadari dampak ini, Jaya Jaya Institut memiliki tujuan krusial untuk dapat mendeteksi siswa yang berisiko tinggi melakukan dropout sedini mungkin, memungkinkan pemberian bimbingan dan intervensi khusus yang tepat waktu. Dalam konteks ini, diperlukan sebuah dashboard yang intuitif, berfungsi sebagai alat bantu utama bagi mereka untuk memahami data siswa dengan mudah dan memonitor performa siswa secara berkelanjutan guna mendukung pengambilan keputusan strategis Jaya Jaya Institut terkait retensi siswa.

### Cakupan Proyek
Proyek data science ini dirancang secara komprehensif untuk mengatasi isu dropout siswa yang dapat berdampak signifikan pada tingkat retensi siswa di Jaya Jaya Institut. Tujuannya tidak hanya untuk memprediksi status akhir siswa (apakah dropout atau tidak) secara akurat, tetapi juga untuk memetakan hubungan dan mengidentifikasi faktor-faktor kunci yang menjadi pendorong fenomena ini menggunakan kekuatan machine learning, khususnya dengan algoritma RandomForest yang dikenal karena robust dan akurat.

Proses analisis dimulai dengan tahapan fundamental exploratory data analysis (EDA) yang mendalam, didukung oleh visualisasi data yang kaya. Tahap awal ini krusial untuk mendapatkan pemahaman awal tentang karakteristik setiap fitur—baik kategorikal maupun numerik—serta secara visual mengaitkan status akhir siswa (dropout/tidak dropout) dengan variabel-variabel penting. Ini mencakup pemeriksaan detail pada Course, Marital_status, Age_at_enrollment, kualifikasi dan pekerjaan orang tua, status displaced, serta kebutuhan khusus, guna mengungkap pola dan tren awal yang mungkin terkait dengan dropout sebelum membangun model prediktif.

Melanjutkan dari pemahaman awal EDA, dilakukan feature engineering terhadap fitur-fitur yang merepresentasikan kinerja siswa selama dua semester pertama. Tujuannya adalah mengonsolidasikan data performa yang bervariasi ini menjadi gambaran metrik kinerja akhir yang lebih ringkas dan representatif untuk setiap siswa, menyediakan fitur yang lebih bermakna dan informatif bagi model Machine Learning.

Setelah itu, data yang sudah melalui rekayasa fitur ini melalui tahap prapemrosesan esensial. Salah satu langkah penting adalah log-transformation yang diterapkan pada kolom-kolom numerik yang menunjukkan skewness tinggi—kondisi di mana distribusi nilai sangat miring ke satu sisi. Transformasi ini membantu menstabilkan varians dan membuat distribusi data lebih simetris, kondisi yang umumnya disukai oleh banyak algoritma machine learning untuk performa optimal.

Untuk mendapatkan pemahaman statistik yang lebih formal dan memvalidasi temuan awal dari EDA, uji statistik juga dilaksanakan. Uji regresi linear multivariate digunakan untuk memeriksa hubungan statistik antara status dropout (sebagai variabel dependen) dengan beberapa fitur kinerja siswa yang bersifat numerik, sementara Uji Chi-square diterapkan untuk menguji independensi atau ada tidaknya asosiasi signifikan antara Status siswa (dropout/tidak dropout) dengan fitur-fitur kategorikal yang telah diidentifikasi relevan pada tahap EDA.

Selanjutnya, serangkaian tahapan prapemprosesan lanjutan dan persiapan model yang lebih spesifik dilakukan. Untuk menangani isu ketidakseimbangan distribusi data antara jumlah siswa dropout (kelas minoritas) dan non-dropout (kelas mayoritas), kami melakukan undersampling pada kelompok mayoritas untuk menyeimbangkan proporsi data pelatihan, mencegah model bias terhadap kelas yang lebih banyak. Kemudian, categorical encoding dengan LabelEncoder diterapkan pada fitur kategorikal target 'Status' untuk mengubahnya menjadi format numerik yang dapat diproses oleh model. Data kemudian dibagi secara acak menjadi 85% untuk pelatihan model dan 15% untuk pengujian performa akhir. Feature scaling menggunakan MinMaxScaler juga diaplikasikan pada kedua set data (pelatihan dan pengujian) untuk menormalisasi rentang nilai fitur numerik ke skala yang seragam, yang penting untuk stabilitas banyak algoritma. Sebagai upaya reduksi dimensi dan untuk meringankan beban komputasi serta potensi mengurangi noise tanpa kehilangan informasi penting, principal component analysis (PCA) diterapkan pada data yang telah diskalakan.

Setelah data siap sepenuhnya dan dalam format yang optimal, model prediktif utama, RandomForest, dilatih menggunakan data pelatihan dan dievaluasi secara independen pada data uji. Untuk memastikan model bekerja pada potensi terbaiknya dan menemukan konfigurasi yang paling efektif, kami mengintegrasikan hyperparameter tuning menggunakan metode RandomizedSearchCV. Proses ini secara efisien mencari kombinasi hyperparameter pelatihan (seperti jumlah pohon, kedalaman pohon, dll.) yang paling optimal guna meningkatkan kinerja prediksi model—baik pada data pelatihan maupun yang belum pernah dilihat sebelumnya—guna menghasilkan model prediksi dropout yang robust dan akurat untuk menginformasikan strategi intervensi.

Proyek data science ini pada dasarnya bergantung pada penggunaan Python sebagai instrumen analisis utama, didukung oleh beragam library esensial yang diperlukan untuk mengelola data serta mengembangkan model machine learning. Seluruh kegiatan analisis ini mutlak memerlukan ketersediaan data performa siswa (data.csv) sebagai sumber masukan utama yang tak terpisahkan. Selain itu, Tableau Public akan dimanfaatkan secara spesifik untuk merancang sebuah business dashboard yang informatif, terdiri dari tiga halaman. Dashboard ini bertujuan untuk menyajikan gambaran umum situasi serta mengeksplorasi faktor-faktor pendorong di balik angka dropout siswa yang cukup tinggi di Jaya Jaya Institut.

Selain komponen utama proyek, terdapat pula file skrip Python terpisah bernama app.py yang fungsi utamanya adalah menjalankan prediksi Status akhir siswa (dropout atau tidak dropout). Skrip ini beroperasi dengan cara memanfaatkan model yang sudah dilatih sebelumnya menggunakan algoritma RandomForest, yang dikembangkan dalam file .ipynb terpisah. Detail lebih lanjut mengenai cara kerja dan eksekusi sistem prediksi ini akan dijelaskan secara komprehensif di bagian 'Menjalankan Sistem Machine Learning' selanjutnya.

### Persiapan

Sumber data: data.csv (https://raw.githubusercontent.com/dicodingacademy/dicoding_dataset/refs/heads/main/students_performance/data.csv)

Setup environment:
```
conda create --name student_dropout_project_submission python=3.11.3
conda activate student_dropout_project_submission
pip install -r requirements.txt

```

## Business Dashboard
Jelaskan tentang business dashboard yang telah dibuat. Jika ada, sertakan juga link untuk mengakses dashboard tersebut.

### Dashboard 1 (Overview of Student Performance Monitoring)
Halaman pertama dashboard ini memberikan gambaran umum penting tentang status dan performa siswa, menyoroti dropout rate signifikan mencapai 32,1% (merepresentasikan 1.421 siswa), yang merupakan angka krusial. Meskipun statistik performa keseluruhan cukup optimis dengan overall approval rate 67,9% dan rata-rata nilai hampir 11, analisis lebih dalam melalui box plot menunjukkan pola menarik: siswa yang drop out ternyata memiliki rentang nilai pada beberapa metrik performa kunci (seperti overall approval rate dan overall average grade) yang tidak jauh berbeda dengan siswa yang masih aktif studi (enrolled). Hal ini mengisyaratkan bahwa penyebab dropout mungkin tidak sepenuhnya tercermin pada metrik performa umum di semester awal. Namun, pada histogram 'grade vs previous qualification difference', terlihat pola yang lebih khas untuk kelompok dropout, dengan mayoritas nilai berada pada rentang 105-140, memberikan petunjuk tambahan tentang karakteristik mereka.

Link URL: https://public.tableau.com/views/StudentDropoutAnalytics/Dashboard1?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link

### Dashboard 2 (Factors related to Student Dropout or No Dropout) 
Dashboard ini menyajikan wawasan penting mengenai profil siswa yang cenderung mengalami dropout, berfokus pada pengaruh faktor kategorikal dan demografis terhadap jumlahnya. Sebuah pola umum yang terlihat jelas di banyak faktor—seperti status pernikahan, keuangan siswa (status hutang, kelancaran SPP, kepemilikan beasiswa), kebutuhan khusus, dan usia 15-25 tahun—adalah bahwa jumlah siswa dropout terbanyak seringkali muncul di kategori yang memiliki jumlah total siswa paling besar, mengisyaratkan angka dropout di sini lebih mencerminkan volume kelompok. Namun, ada nuansa dan indikator risiko spesifik seperti beberapa kelas (course) tertentu (contohnya Management (evening), Equinculture, Informatics Engineering) yang menunjukkan tingkat dropout sangat tinggi (>50%). Selain itu, visualisasi menunjukkan jumlah dropout lebih tinggi pada siswa yang tidak displaced dibandingkan yang displaced, dan analisis kebangsaan mengungkap siswa internasional memiliki dropout rate yang lebih tinggi (32,2%) dibandingkan siswa nasional (29,1%), menyoroti perbedaan proporsi risiko yang signifikan yang memerlukan perhatian lebih lanjut.

Link URL: https://public.tableau.com/views/StudentDropoutAnalytics/Dashboard2?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link 

## Menjalankan Sistem Machine Learning
Jelaskan cara menjalankan protoype sistem machine learning yang telah dibuat. Selain itu, sertakan juga link untuk mengakses prototype tersebut.

Untuk menjalankan prototype machine learning ini, pengguna pertama-tama memasukkan data melalui antarmuka Streamlit. Input ini mencakup informasi seperti 'Usia saat pendaftaran', 'Application_order', 'Admission grade', nilai kualifikasi sebelumnya, serta rincian unit kurikuler untuk semester pertama dan kedua (jumlah unit kurikuler yang diambil, terdaftar, dievaluasi, disetujui, nilai, dan yang tanpa evaluasi). Setelah semua data dimasukkan, pengguna mengklik tombol "Prediksi data Anda". Tindakan ini memicu pengumpulan data input, yang kemudian diproses oleh fungsi transform_data. Fungsi ini melakukan pra-pemrosesan pada data input dengan memanggil preprocess_data untuk membuat fitur-fitur baru, sekaligus memproses data pelatihan (diambil langsung dari GitHub raw URL, dibersihkan, dilakukan feature engineering, skewed variables dilakukan log-transformation, dilakukan undersampling, kolom tidak relevan dihapus, dan fitur kategorikal di-encode), melakukan penskalaan menggunakan MinMaxScaler yang telah di-fit pada data latih, dan menerapkan transformasi PCA. Selanjutnya, fungsi predict() dipanggil, yang memuat model model.joblib (model RandomForest yang sudah dilatih) untuk memprediksi status akhir mahasiswa ('Dropout', 'Enrolled', atau 'Graduate') berdasarkan data input yang telah ditransformasi. Hasil prediksi status kemudian ditampilkan kepada pengguna, bersama dengan tabel data pra-pemrosesan awal dalam bagian expander "View preprocessed data" 

```
asasaadadadada

```

## Conclusion
Proyek ini dirancang untuk mengatasi isu dropout siswa di Jaya Jaya Institut dengan tujuan memprediksi status akhir siswa secara akurat dan mengidentifikasi faktor pendorong utamanya menggunakan Machine Learning, terutama algoritma RandomForest. Proses ini meliputi tahapan komprehensif seperti EDA untuk memahami data dan mengaitkan fitur kunci (usia, status, latar belakang, dll.) dengan dropout, feature engineering, serta prapemrosesan data (transformasi, penanganan ketidakseimbangan dengan undersampling, penskalaan, PCA) sebelum melatih model. Uji statistik seperti Regresi dan Chi-square juga dilakukan untuk validasi awal. Proyek ini menghasilkan model prediksi RandomForest yang dilatih dengan hyperparameter tuning, dashboard monitoring berbasis Tableau, dan prototype prediksi individual berbasis Streamlit, kesemuanya dibangun menggunakan Python dan library terkait, mengandalkan data performa siswa sebagai input utama.

Business dashboard yang telah dibangun sebelumnya mengilustrasikan gambaran yang jelas tentang tantangan dropout di Jaya Jaya Institut. Realitas pertama adalah dropout rate yang signifikan mencapai 32,1%, melibatkan 1.421 siswa—sebuah jumlah yang krusial. Menariknya, analisis performa awal menunjukkan bahwa metrik umum siswa dropout tidak jauh berbeda dengan mereka yang masih enrolled, menyiratkan isu lebih dalam dari sekadar nilai semata. Data justru menyoroti faktor pendorong spesifik: siswa yang berisiko adalah mereka yang berusia 15-25 tahun, berstatus lajang, dengan latar belakang orang tua/kualifikasi awal tertentu, atau isu keuangan. Lebih lanjut, beberapa kelas/jurusan menunjukkan rate dropout di atas 50%, siswa non-displaced mencatat jumlah dropout lebih tinggi, dan mahasiswa internasional memiliki rate dropout yang lebih tinggi (32,2%) dibanding nasional. Temuan ini memberikan peta yang jelas tentang siapa yang rentan dan mengapa, memandu langkah strategis selanjutnya.

### Rekomendasi Action Items
Berikan beberapa rekomendasi action items yang harus dilakukan perusahaan guna menyelesaikan permasalahan atau mencapai target mereka.
- action item 1
- action item 2