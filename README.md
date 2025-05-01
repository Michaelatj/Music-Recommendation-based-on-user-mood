# Aplikasi Rekomendasi Musik Berdasarkan Mood Pengguna

Aplikasi ini membantu merekomendasikan lagu sesuai mood yang dipilih pengguna. Sistem memanfaatkan model machine learning yang sudah dilatih untuk mengklasifikasikan lagu ke dalam beberapa kategori mood seperti Happy, Sad, Anger, Focused, Romantic, Neutral, dan Relaxed.

## Link Demo Aplikasi

Aplikasi sudah berhasil dideploy dan dapat langsung dicoba melalui link berikut:  
👉 **[https://alongbest2000.pythonanywhere.com/](https://alongbest2000.pythonanywhere.com/)**

## Struktur Proyek

```
Music-Recommendation-based-on-user-mood
├── app.py                      # Script utama aplikasi Flask
├── Procfile                    # Konfigurasi deployment (misal: PythonAnywhere)
├── README.md                   # Dokumentasi proyek
├── requirements.txt            # Daftar dependensi Python
├── tempCodeRunnerFile.py       # File sementara (dapat diabaikan)
├── training_model.ipynb        # Notebook training & evaluasi model
├── .ipynb_checkpoints/         # Folder checkpoint Jupyter (otomatis)
│   └── Untitled-checkpoint.ipynb
├── dataset/        
│   ├── cleaned_data.csv        # Dataset hasil pembersihan
│   └── music_data.csv          # Dataset utama lagu & fitur audio
├── models/
│   ├── mood_model.joblib       # Model klasifikasi mood (hasil training)
│   └── mood_scaler.joblib      # Scaler untuk normalisasi fitur
├── static/
│   ├── css/
│   │   └── style.css           # CSS untuk tampilan aplikasi
│   └── icons/
│       ├── anger.png
│       ├── focused.png
│       ├── happy.png
│       ├── neutral.png
│       ├── relaxed.png
│       ├── romantic.png
│       ├── sad.png
│       └── spotify_icon.png
└── templates/
    ├── index.html              # Halaman pemilihan mood
    └── recommendations.html    # Halaman hasil rekomendasi
```

## Fitur Utama

- **Rekomendasi Lagu Berdasarkan Mood:**  
  Pilih mood, aplikasi akan menampilkan daftar lagu yang sesuai.
- **Pencarian Lagu:**  
  Cari lagu berdasarkan nama pada halaman rekomendasi.
- **Filter Berdasarkan Huruf Awal:**  
  Filter lagu berdasarkan huruf pertama judul.
- **Integrasi Spotify:**  
  Setiap lagu memiliki tautan langsung ke Spotify.
- **Tampilan Responsif & Ikon Mood:**  
  UI modern dengan ikon mood dan Spotify.

## Cara Menjalankan Aplikasi

1. **Clone repository**
   ```bash
   git clone https://github.com/Michaelatj/Music-Recommendation-based-on-user-mood.git
   cd Music-Recommendation-based-on-user-mood
   ```

2. **Install dependensi**
   ```bash
   pip install -r requirements.txt
   ```

3. **Pastikan file model sudah ada di folder `models/`**
   
   Jika belum ada, jalankan notebook `training_model.ipynb` untuk membuat model dan scaler.

4. **Jalankan aplikasi**
   ```bash
   python app.py
   ```
   Buka aplikasi di browser: [http://127.0.0.1:5000](http://127.0.0.1:5000)

## Penjelasan File Penting

- [`app.py`](app.py):  
  Script utama Flask, menangani routing, load model, prediksi mood, rekomendasi lagu, search, dan filter.
- [`training_model.ipynb`](training_model.ipynb):  
  Notebook untuk pembersihan data, rekayasa fitur, training, evaluasi, dan penyimpanan model.
- [`models/mood_model.joblib`](models/mood_model.joblib) & [`models/mood_scaler.joblib`](models/mood_scaler.joblib):  
  Model dan scaler hasil training.
- [`dataset/music_data.csv`](dataset/music_data.csv):  
  Dataset utama lagu dan fitur audio.
- [`templates/index.html`](templates/index.html) & [`templates/recommendations.html`](templates/recommendations.html):  
  Template HTML untuk UI aplikasi.
- [`static/css/style.css`](static/css/style.css):  
  CSS untuk styling aplikasi.

## Catatan

- File `training_model.ipynb` hanya dijalankan saat ingin training ulang model (misal: update dataset atau tuning model).
- Untuk deployment di PythonAnywhere atau platform lain, gunakan `Procfile` dan pastikan semua dependensi sudah terinstall.
- File/folder seperti `tempCodeRunnerFile.py`, `.ipynb_checkpoints/` hanya untuk keperluan development dan bisa diabaikan saat produksi.

## Daftar Anggota & Kontribusi

| No | Nama Lengkap                | NIM         | Kontribusi Utama                                                                 |
|----|-----------------------------|-------------|----------------------------------------------------------------------------------|
| 1  | Michael Andreas Tjendra     | 231111210   | Training & evaluasi model, penyimpanan model & scaler                            |
| 2  | Silvani Chayadi             | 231112945   | Backend Flask: rekomendasi, search, filter, API routing                          |
| 3  | Diky Diwa Suwanto           | 231110760   | Frontend (HTML, CSS), integrasi web, demo aplikasi                               |
| 4  | Reno Kurniawan Panjaitan    | 231112553   | Pembersihan data, rekayasa fitur, pelabelan mood pada dataset                    |
| 5  | Cindy Nathania              | 231111567   | Backend Flask: load model, preprocessing data, prediksi mood                     |

---

**Selamat mencoba aplikasi rekomendasi musik berbasis mood!**
