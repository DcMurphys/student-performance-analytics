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

Selanjutnya, serangkaian tahapan prapemprosesan lanjutan dan persiapan model yang lebih spesifik dilakukan. Untuk menangani isu ketidakseimbangan distribusi data antara jumlah siswa dropout (kelas minoritas) dan non-dropout (kelas mayoritas), kita melakukan undersampling pada kelompok mayoritas untuk menyeimbangkan proporsi data pelatihan, mencegah model bias terhadap kelas yang lebih banyak. Kemudian, categorical encoding dengan LabelEncoder diterapkan pada fitur kategorikal target 'Status' untuk mengubahnya menjadi format numerik yang dapat diproses oleh model. Data kemudian dibagi secara acak menjadi 90% untuk pelatihan model dan 10% untuk pengujian performa akhir. Feature scaling menggunakan MinMaxScaler juga diaplikasikan pada kedua set data (pelatihan dan pengujian) untuk menormalisasi rentang nilai fitur numerik ke skala yang seragam, yang penting untuk stabilitas banyak algoritma. Sebagai upaya reduksi dimensi dan untuk meringankan beban komputasi serta potensi mengurangi noise tanpa kehilangan informasi penting, principal component analysis (PCA) diterapkan pada data yang telah diskalakan.

Setelah data siap sepenuhnya dan dalam format yang optimal, model prediktif utama, RandomForest, dilatih menggunakan data pelatihan dan dievaluasi secara independen pada data uji. Untuk memastikan model bekerja pada potensi terbaiknya dan menemukan konfigurasi yang paling efektif, kita mengintegrasikan hyperparameter tuning menggunakan metode RandomizedSearchCV. Proses ini secara efisien mencari kombinasi hyperparameter pelatihan (seperti jumlah pohon, kedalaman pohon, dll.) yang paling optimal guna meningkatkan kinerja prediksi model—baik pada data pelatihan maupun yang belum pernah dilihat sebelumnya—guna menghasilkan model prediksi dropout yang robust dan akurat untuk menginformasikan strategi intervensi.

Proyek data science ini pada dasarnya bergantung pada penggunaan Python sebagai instrumen analisis utama, didukung oleh beragam library esensial yang diperlukan untuk mengelola data serta mengembangkan model machine learning. Seluruh kegiatan analisis ini mutlak memerlukan ketersediaan data performa siswa (data.csv) sebagai sumber masukan utama yang tak terpisahkan. Selain itu, Tableau Public akan dimanfaatkan secara spesifik untuk merancang sebuah business dashboard yang informatif, terdiri dari dua halaman. Dashboard ini bertujuan untuk menyajikan gambaran umum situasi serta mengeksplorasi faktor-faktor pendorong di balik angka dropout siswa yang cukup tinggi di Jaya Jaya Institut.

Selain komponen utama proyek, terdapat pula file skrip Python terpisah bernama app.py yang fungsi utamanya adalah menjalankan prediksi Status akhir siswa (dropout atau tidak dropout). Skrip ini beroperasi dengan cara memanfaatkan model yang sudah dilatih sebelumnya menggunakan algoritma RandomForest serta variabel target 'Status' yang diperoleh dari file label_encoder.joblib, yang dikembangkan dalam file .ipynb terpisah. Cara kerja dari sistem prediksi ini adalah dengan melakukan prediksi label 'Status' akhir siswa pada input data yang sudah diberikan, di mana input data ini sudah melalui proses feature engineering, encoding, scaling, dan PCA reduction.

### Persiapan

Sumber data: [Student's performance](https://raw.githubusercontent.com/dicodingacademy/dicoding_dataset/refs/heads/main/students_performance/data.csv)

Setup environment:

```
conda create --name student_dropout_project_submission python=3.11.3
conda activate student_dropout_project_submission
pip install -r requirements.txt

```

## Business Dashboard

### Dashboard 1 (Overview of Student Performance Monitoring)

Halaman pertama dashboard ini memberikan gambaran umum penting tentang status dan performa siswa, menyoroti dropout rate signifikan mencapai 32,1% (merepresentasikan 1.421 siswa), yang merupakan angka krusial. Meskipun statistik performa keseluruhan cukup optimis dengan overall approval rate 67,9% dan rata-rata nilai hampir 11, analisis lebih dalam melalui box plot menunjukkan pola menarik: siswa yang drop out ternyata memiliki rentang nilai pada beberapa metrik performa kunci (seperti overall approval rate dan overall average grade) yang tidak jauh berbeda dengan siswa yang masih aktif studi (enrolled). Hal ini mengisyaratkan bahwa penyebab dropout mungkin tidak sepenuhnya tercermin pada metrik performa umum di semester awal. Namun, pada histogram 'grade vs previous qualification difference', terlihat pola yang lebih khas untuk kelompok dropout, dengan mayoritas nilai berada pada rentang 105-140, memberikan petunjuk tambahan tentang karakteristik mereka.

Link URL: [Dashboard 1 via Tableau Public](https://public.tableau.com/views/StudentDropoutAnalytics/Dashboard1?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)

### Dashboard 2 (Factors related to Student Dropout or No Dropout)

Dashboard ini menyajikan wawasan penting mengenai profil siswa yang cenderung mengalami dropout, berfokus pada pengaruh faktor kategorikal dan demografis terhadap jumlahnya. Sebuah pola umum yang terlihat jelas di banyak faktor—seperti status pernikahan, keuangan siswa (status hutang, kelancaran uang kuliah tunggal, kepemilikan beasiswa), kebutuhan khusus, dan usia 15-25 tahun—adalah bahwa jumlah siswa dropout terbanyak seringkali muncul di kategori yang memiliki jumlah total siswa paling besar, mengisyaratkan angka dropout di sini lebih mencerminkan volume kelompok. Namun, ada nuansa dan indikator risiko spesifik seperti beberapa kelas (course) tertentu (contohnya Management (evening), Equinculture, Informatics Engineering) yang menunjukkan tingkat dropout sangat tinggi (>50%). Selain itu, visualisasi menunjukkan jumlah dropout lebih tinggi pada siswa yang tidak displaced dibandingkan yang displaced, dan analisis kebangsaan mengungkap siswa internasional memiliki dropout rate yang lebih tinggi (32,2%) dibandingkan siswa nasional (29,1%), menyoroti perbedaan proporsi risiko yang signifikan yang memerlukan perhatian lebih lanjut.

Link URL: [Dashboard 2 via Tableau Public](https://public.tableau.com/views/StudentDropoutAnalytics/Dashboard2?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)

## Menjalankan Sistem Machine Learning

Ada dua cara yang bisa kita lakukan untuk mencoba menjalankan prototype sistem machine learning dalam proyek kali ini, yaitu menjalankannya secara lokal maupun secara online (di-deploy di cloud lewat Streamlit Cloud). Berikut ini adalah tahapan dalam menjalankan prototype sistem machine learning dalam dua cara.

**Menjalankan prototype machine learning secara lokal**

1. Buka terminal (misalnya command prompt) dan aktifkan virtual environment yang sudah kita buat sebelumnya dengan prompt berikut

```
conda activate student_dropout_project_submission

```

2. Pastikan direktori yang menjadi tempat penyimpanan program prototype machine learning (misalnya 'student_dropout_project_submission' dalam proyek kali ini) sudah berisikan file-file yang diperlukan, terutama file app.py dan requirements.txt (library dependencies) untuk menjalankan sistem machine learning. Apabila kita masih belum berada pada direktori tersebut, kita bisa mengarahkan command prompt dengan prompt berikut

```
cd path/to/destination/directory

```

3. Apabila sudah berada pada direktori yang sesuai, jalankan prompt berikut ini untuk membuka aplikasi app.py di lokal melalui Streamlit

```
streamlit run app.py

```

4. Apabila file tersebut berhasil dijalankan tanpa adanya error sama sekali, kita bisa mengisikan data yang diperlukan dan kemudian mengklik tombol 'Predict your data' untuk mendapatkan status akhir dari siswa (Dropout atau tidak)

**Menjalankan prototype machine learning di cloud (online)**

1. Buka browser kegemaran teman-teman semua
2. Akses link berikut ini: [Student Status Prediction App](https://dcmurphys-student-performance-analytics.streamlit.app)
3. Apabila link ini bisa dijalankan tanpa adanya error sama sekali, kita bisa langsung mengisikan data yang diperlukan dan kemudian mengklik tombol 'Predict your data' untuk mendapatkan status akhir dari siswa (Dropout atau tidak)

## Conclusion

Proyek ini dirancang untuk mengatasi isu dropout siswa di Jaya Jaya Institut dengan tujuan memprediksi status akhir siswa secara akurat dan mengidentifikasi faktor pendorong utamanya menggunakan Machine Learning, terutama algoritma RandomForest. Proses ini meliputi tahapan komprehensif seperti EDA untuk memahami data dan mengaitkan fitur kunci (usia, status, latar belakang, dll.) dengan dropout, feature engineering, serta prapemrosesan data (transformasi, penanganan ketidakseimbangan dengan undersampling, penskalaan, PCA) sebelum melatih model. Uji statistik seperti Regresi dan Chi-square juga dilakukan untuk validasi awal. Proyek ini menghasilkan model prediksi RandomForest yang dilatih dengan hyperparameter tuning, dashboard monitoring berbasis Tableau, dan prototype prediksi individual berbasis Streamlit, kesemuanya dibangun menggunakan Python dan library terkait, mengandalkan data performa siswa sebagai input utama.

Business dashboard yang telah dibangun sebelumnya mengilustrasikan gambaran yang jelas tentang tantangan dropout di Jaya Jaya Institut. Realitas pertama adalah dropout rate yang signifikan mencapai 32,1%, melibatkan 1.421 siswa—sebuah jumlah yang krusial. Menariknya, analisis performa awal menunjukkan bahwa metrik umum siswa dropout tidak jauh berbeda dengan mereka yang masih enrolled, menyiratkan isu lebih dalam dari sekadar nilai semata. Data justru menyoroti faktor pendorong spesifik: siswa yang berisiko adalah mereka yang berusia 15-25 tahun, berstatus lajang, dengan latar belakang orang tua/kualifikasi awal tertentu, atau isu keuangan. Lebih lanjut, beberapa kelas/jurusan menunjukkan rate dropout di atas 50%, siswa non-displaced mencatat jumlah dropout lebih tinggi, dan mahasiswa internasional memiliki rate dropout yang lebih tinggi (32,2%) dibanding nasional. Temuan ini memberikan peta yang jelas tentang siapa yang rentan dan mengapa, memandu langkah strategis selanjutnya.

### Rekomendasi Action Items

Berdasarkan temuan yang diperoleh dari business dashboard yang telah dibangun terkait jumlah siswa yang dropout beserta karakteristik dan faktor-faktornya yang mendasari hal tersebut, Jaya Jaya Institut perlu mempertimbangkan beberapa langkah strategis berikut dalam rangka meminimalisasi angka dropout dan mempertahankan tingkat retensi siswanya berdasarkan saran dari Nurmalitasari et Al. (2023):

- Jaya Jaya Institut perlu memperkuat program bantuan finansial, beasiswa, dan peluang kerja paruh waktu yang fleksibel sebagai tindak lanjut dari temuan isu finansial pada siswa dropout. Penting juga untuk menyediakan konseling pengelolaan keuangan bagi siswa yang rentan.
- Jaya Jaya Institut sebaiknya meningkatkan kualitas pengajaran, interaksi dosen-siswa, serta layanan konseling akademis yang lebih kuat untuk mengatasi isu performa (terutama perbedaan nilai kualifikasi awal vs saat ini) dan tingginya dropout di kelas/jurusan tertentu. Panduan dalam pemilihan program studi yang sesuai minat/kemampuan siswa juga perlu ditingkatkan. Selain itu, dukungan khusus untuk mata kuliah sulit atau aspek seperti penyelesaian tugas akhir juga krusial.
- Jaya Jaya Institut perlu mengembangkan program orientasi, mentorship, dan komunitas yang lebih kuat untuk siswa di rentang usia 15-25 tahun dan siswa lajang.
- Layanan dukungan spesifik (akademis, sosial, kultural) yang disesuaikan untuk mahasiswa internasional juga perlu disediakan mengingat dropout rate mereka yang lebih tinggi.
- Analisis diagnostik lebih lanjut terhadap akar masalah tingginya dropout pada siswa non-displaced perlu dilakukan agar intervensi terhadap isu ini dapat dirancang dengan tepat.
- Pihak institusi perlu memanfaatkan data dari dashboard dan analisis (termasuk model prediksi) untuk mengidentifikasi siswa berisiko secara proaktif, dan menyalurkan berbagai bentuk dukungan (finansial, akademis, bimbingan, dll.) kepada siswa-siswa ini sesegera mungkin, berdasarkan temuan profil risiko yang paling menonjol.

### Referensi

Nurmalitasari, Awang Long, Z., & Faizuddin Mohd Noor, M. (2023). Factors influencing dropout students in higher education. _Education Research International_, 2023(1), 7704142.
