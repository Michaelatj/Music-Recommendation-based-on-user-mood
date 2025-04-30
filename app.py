# --- Import Library ---
import os
import pandas as pd
import joblib
from flask import Flask, render_template, request, url_for, jsonify

# --- Flask App Setup ---
app = Flask(__name__)

# --- Path Configurations ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'mood_model.joblib')
SCALER_PATH = os.path.join(BASE_DIR, 'models', 'mood_scaler.joblib')
DATA_PATH = os.path.join(BASE_DIR, 'dataset', 'cleaned_data.csv')

# --- Constants ---
FEATURES = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness',
            'instrumentalness', 'liveness', 'valence', 'tempo', 'mode', 'duration_ms']
AVAILABLE_MOODS = ['Happy', 'Sad', 'Anger', 'Focused', 'Romantic', 'Neutral', 'Relaxed']
INITIAL_LOAD_COUNT = 10
LOAD_MORE_COUNT = 10

# --- Load Model, Scaler, and Dataset ---
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
except FileNotFoundError as e:
    print(f"Error: Model or scaler file not found: {e}")
    exit()

try:
    df_full = pd.read_csv(DATA_PATH)
    if 'Unnamed: 0' in df_full.columns:
        df_full = df_full.drop('Unnamed: 0', axis=1)

    # Validasi kolom fitur dan kolom penting lainnya
    missing_features = [f for f in FEATURES if f not in df_full.columns]
    if missing_features:
        raise ValueError(f"Missing features: {missing_features}")

    required_cols = ['track_id', 'track_name', 'artists', 'album_name', 'track_genre', 'popularity']
    missing_required = [col for col in required_cols if col not in df_full.columns]
    if missing_required:
        raise ValueError(f"Missing columns: {missing_required}")

    # Penanganan missing values dan prediksi mood
    df_full[FEATURES] = df_full[FEATURES].fillna(df_full[FEATURES].median())
    music_features_scaled = scaler.transform(df_full[FEATURES])
    df_full['predicted_mood'] = model.predict(music_features_scaled)

    # Mapping jika label model bukan string mood
    mood_mapping = {
        # Contoh: '0': 'Happy'
    }
    df_full['predicted_mood'] = df_full['predicted_mood'].map(mood_mapping).fillna(df_full['predicted_mood'])

    # Dataset unik berdasarkan track_id, disortir berdasarkan popularitas
    all_unique_songs = df_full.drop_duplicates(subset=['track_id'], keep='first')
    all_unique_songs_sorted = all_unique_songs.sort_values(by='popularity', ascending=False)

except Exception as e:
    print(f"Data processing error: {e}")
    exit()

# --- Recommendation Function ---
def get_recommendations_slice(mood_input, offset=0, limit=10):
    filtered_songs = all_unique_songs_sorted[all_unique_songs_sorted['predicted_mood'] == mood_input]
    if filtered_songs.empty:
        filtered_songs = all_unique_songs_sorted  # fallback
    result = filtered_songs.iloc[offset : offset + limit]
    cols = ['track_name', 'artists', 'album_name', 'track_genre', 'popularity', 'track_id']
    return result[cols], len(filtered_songs)

# --- Search Function by Title ---
def search_songs(query, offset=0, limit=10):
    query = query.strip().lower()
    result = all_unique_songs_sorted[
        all_unique_songs_sorted['track_name'].str.lower().str.startswith(query, na=False)
    ]
    return result.iloc[offset : offset + limit], len(result)

# --- Filter by First Letter ---
def filter_by_letter(letter, offset=0, limit=10):
    letter = letter.strip().lower()
    result = all_unique_songs_sorted[
        all_unique_songs_sorted['track_name'].str.lower().str.startswith(letter, na=False)
    ]
    if result.empty:
        result = all_unique_songs_sorted  # fallback
    cols = ['track_name', 'artists', 'album_name', 'track_genre', 'popularity', 'track_id']
    return result.iloc[offset : offset + limit][cols], len(result)

# --- Web Routes ---
@app.route('/')
def index():
    return render_template('index.html', moods=AVAILABLE_MOODS)

@app.route('/recommend', methods=['POST'])
def recommend():
    selected_mood = request.form.get('mood')
    if not selected_mood or selected_mood not in AVAILABLE_MOODS:
        return render_template('recommendations.html', error="Invalid mood selected.")

    result_df, total = get_recommendations_slice(selected_mood, 0, INITIAL_LOAD_COUNT)
    show_more = total > INITIAL_LOAD_COUNT
    return render_template('recommendations.html',
                           selected_mood=selected_mood,
                           recommendations=result_df,
                           show_more=show_more,
                           initial_count=INITIAL_LOAD_COUNT,
                           load_more_count=LOAD_MORE_COUNT)

# --- API Endpoint: Get Mood-Based Recommendations ---
@app.route('/api/recommendations')
def api_recommendations():
    mood = request.args.get('mood')
    offset = int(request.args.get('offset', 0))
    if not mood or mood not in AVAILABLE_MOODS:
        return jsonify({"error": "Invalid or unknown mood"}), 400
    result_df, total = get_recommendations_slice(mood, offset, LOAD_MORE_COUNT)
    return jsonify({
        "recommendations": result_df.to_dict(orient='records'),
        "has_more": offset + len(result_df) < total
    })

# --- API Endpoint: Search by Title Query ---
@app.route('/api/search')
def api_search():
    query = request.args.get('query', '')
    offset = int(request.args.get('offset', 0))
    if not query:
        return jsonify({"error": "Query is required"}), 400
    result_df, total = search_songs(query, offset, LOAD_MORE_COUNT)
    if result_df.empty:
        return jsonify({"error": f"No songs found starting with '{query}'"})
    return jsonify({
        "recommendations": result_df.to_dict(orient='records'),
        "has_more": offset + len(result_df) < total
    })

# --- API Endpoint: Filter Songs by Starting Letter ---
@app.route('/api/filter_by_letter')
def api_filter_by_letter():
    letter = request.args.get('letter', '')
    offset = int(request.args.get('offset', 0))
    if not letter or len(letter) != 1 or not letter.isalpha():
        return jsonify({"error": "Letter must be a single alphabet character"}), 400
    result_df, total = filter_by_letter(letter, offset, LOAD_MORE_COUNT)
    return jsonify({
        "recommendations": result_df.to_dict(orient='records'),
        "has_more": offset + len(result_df) < total
    })

# --- Run App ---
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
