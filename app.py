import os
import pandas as pd
import joblib
# Tambahkan jsonify untuk rute API
from flask import Flask, render_template, request, url_for, jsonify

# --- Konfigurasi dan Pemuatan Aset ---
app = Flask(__name__)

# Path relatif ke aset (lebih robust)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'mood_model.joblib')
SCALER_PATH = os.path.join(BASE_DIR, 'models', 'mood_scaler.joblib')
# Pastikan path dataset benar dan folder 'data' ada (bukan 'dataset')
# Jika folder Anda memang 'dataset', ganti 'data' di bawah ini menjadi 'dataset'
DATA_PATH = os.path.join(BASE_DIR, 'data', 'C:/Users/along/OneDrive/Desktop/coding/AI Project/Music Recommendation base on user mood/dataset/music_data.csv') # <-- Pastikan NAMA FILE & FOLDER CSV BENAR

FEATURES = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness',
            'instrumentalness', 'liveness', 'valence', 'tempo', 'mode', 'duration_ms']
AVAILABLE_MOODS = ['Happy', 'Sad', 'Anger', 'Focused', 'Romantic', 'Neutral', 'Relaxed']
INITIAL_LOAD_COUNT = 10 # Jumlah lagu yang ditampilkan awal
LOAD_MORE_COUNT = 10  # Jumlah lagu yang dimuat saat "Show More"

# --- Pemuatan Model, Scaler, dan Data (Sama seperti sebelumnya) ---
# ... (kode pemuatan model, scaler, dan df_full tetap sama) ...
# Pastikan bagian ini berhasil memuat df_full dan menambahkan 'predicted_mood'
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("Model dan Scaler berhasil dimuat.")
except FileNotFoundError:
    print(f"Error: File model atau scaler tidak ditemukan.")
    exit()

try:
    df_full = pd.read_csv(DATA_PATH)
    print(f"Dataset '{DATA_PATH}' berhasil dimuat. Jumlah baris awal: {len(df_full)}")
    if 'Unnamed: 0' in df_full.columns:
        df_full = df_full.drop('Unnamed: 0', axis=1)

    missing_features = [f for f in FEATURES if f not in df_full.columns]
    if missing_features:
         raise ValueError(f"Fitur berikut hilang dari dataset: {missing_features}")

    required_cols_exist = ['track_id', 'track_name', 'artists', 'album_name', 'track_genre', 'popularity']
    missing_required = [col for col in required_cols_exist if col not in df_full.columns]
    if missing_required:
        raise ValueError(f"Kolom wajib berikut hilang dari dataset: {missing_required}")


    df_full[FEATURES] = df_full[FEATURES].fillna(df_full[FEATURES].median())
    music_features_scaled = scaler.transform(df_full[FEATURES])
    df_full['predicted_mood'] = model.predict(music_features_scaled)
    print("Prediksi mood untuk seluruh dataset selesai.")
    print("Distribusi mood yang diprediksi:")
    print(df_full['predicted_mood'].value_counts())

    # Pra-filter dan pra-urutkan sekali untuk efisiensi
    all_unique_songs = df_full.drop_duplicates(subset=['track_id'], keep='first')
    all_unique_songs_sorted = all_unique_songs.sort_values(by='popularity', ascending=False)
    print("Dataset unik telah diurutkan berdasarkan popularitas.")

except FileNotFoundError:
    print(f"Error: File dataset tidak ditemukan di '{DATA_PATH}'.")
    exit()
except ValueError as ve:
    print(f"Error saat memproses dataset: {ve}")
    exit()
except Exception as e:
     print(f"Terjadi error tak terduga saat memuat/memproses data: {e}")
     exit()


# --- Fungsi Rekomendasi Diperbarui ---
def get_recommendations_slice(mood_input, offset=0, limit=10):
    """Mengambil slice rekomendasi musik unik berdasarkan mood."""
    # Gunakan dataframe yang sudah diurutkan dan unik
    filtered_songs = all_unique_songs_sorted[all_unique_songs_sorted['predicted_mood'] == mood_input]

    # Hitung total lagu yang cocok untuk mood ini
    total_matching_songs = len(filtered_songs)

    # Ambil slice berdasarkan offset dan limit
    recommendations_slice = filtered_songs.iloc[offset : offset + limit]

    # Kolom yang dibutuhkan di output (termasuk track_id untuk link Spotify)
    required_display_cols = ['track_name', 'artists', 'album_name', 'track_genre', 'popularity', 'track_id']

    # Cek apakah kolom yang dibutuhkan ada (seharusnya sudah dicek saat load)
    missing_display_cols = [col for col in required_display_cols if col not in recommendations_slice.columns]
    if missing_display_cols:
         # Ini seharusnya tidak terjadi jika pengecekan awal berhasil
         print(f"Peringatan: Kolom display berikut hilang saat slicing: {missing_display_cols}")
         # Kembalikan data yang ada saja atau handle error
         existing_cols = [col for col in required_display_cols if col in recommendations_slice.columns]
         return recommendations_slice[existing_cols], total_matching_songs

    return recommendations_slice[required_display_cols], total_matching_songs


# --- Rute Flask ---
@app.route('/')
def index():
    """Menampilkan halaman utama."""
    return render_template('index.html', moods=AVAILABLE_MOODS)

@app.route('/recommend', methods=['POST'])
def recommend():
    """Menampilkan halaman rekomendasi AWAL."""
    selected_mood = request.form.get('mood')
    recommendations_result = None
    error_message = None
    total_songs_for_mood = 0

    if not selected_mood:
        error_message = "Silakan pilih salah satu mood."
    elif selected_mood not in AVAILABLE_MOODS:
         error_message = f"Mood '{selected_mood}' tidak valid."
    else:
        # Ambil batch AWAL (offset=0, limit=INITIAL_LOAD_COUNT)
        result_df, total_songs_for_mood = get_recommendations_slice(
            selected_mood,
            offset=0,
            limit=INITIAL_LOAD_COUNT
        )

        if not result_df.empty:
            recommendations_result = result_df
        elif total_songs_for_mood == 0:
             # Jika tidak ada sama sekali lagu untuk mood ini
            if selected_mood not in df_full['predicted_mood'].unique():
                 error_message = f"Maaf, model saat ini tidak menghasilkan prediksi untuk mood '{selected_mood}'. Coba mood lain."
            else:
                error_message = f"Tidak ditemukan lagu yang cocok untuk mood '{selected_mood}'."
        # else: Kondisi result_df kosong tapi total > 0 tidak mungkin dengan slicing iloc

    # Hitung apakah ada lebih banyak lagu untuk ditampilkan
    show_more_possible = total_songs_for_mood > INITIAL_LOAD_COUNT

    return render_template('recommendations.html',
                           selected_mood=selected_mood,
                           recommendations=recommendations_result,
                           error=error_message,
                           show_more=show_more_possible, # Kirim flag ini ke template
                           initial_count=INITIAL_LOAD_COUNT, # Kirim jumlah awal
                           load_more_count=LOAD_MORE_COUNT # Kirim jumlah per load more
                           )

# --- Rute API BARU untuk "Show More" ---
@app.route('/api/recommendations')
def api_recommendations():
    """Mengembalikan data rekomendasi berikutnya sebagai JSON."""
    mood = request.args.get('mood')
    # Dapatkan offset (jumlah lagu yang sudah ditampilkan)
    try:
        offset = int(request.args.get('offset', 0))
    except ValueError:
        return jsonify({"error": "Invalid offset value"}), 400

    if not mood or mood not in df_full['predicted_mood'].unique():
        return jsonify({"error": "Invalid or unknown mood"}), 400

    # Ambil batch BERIKUTNYA
    limit = LOAD_MORE_COUNT
    result_df, total_songs_for_mood = get_recommendations_slice(mood, offset=offset, limit=limit)

    # Konversi ke format JSON yang mudah digunakan JavaScript (list of dicts)
    recommendations_json = result_df.to_dict(orient='records')

    # Tentukan apakah ada lebih banyak lagu setelah batch ini
    has_more = (offset + len(recommendations_json)) < total_songs_for_mood

    return jsonify({
        "recommendations": recommendations_json,
        "has_more": has_more # Kirim info ini ke frontend
    })


# --- Menjalankan Aplikasi ---
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)