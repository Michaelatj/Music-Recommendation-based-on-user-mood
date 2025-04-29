import os
import pandas as pd
import joblib
from flask import Flask, render_template, request, url_for, jsonify

# --- Configuration and Asset Loading ---
app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'mood_model.joblib')
SCALER_PATH = os.path.join(BASE_DIR, 'models', 'mood_scaler.joblib')
DATA_PATH = os.path.join(BASE_DIR, 'dataset', 'music_data.csv')

FEATURES = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness',
            'instrumentalness', 'liveness', 'valence', 'tempo', 'mode', 'duration_ms']
AVAILABLE_MOODS = ['Happy', 'Sad', 'Anger', 'Focused', 'Romantic', 'Neutral', 'Relaxed']
INITIAL_LOAD_COUNT = 10
LOAD_MORE_COUNT = 10

# --- Load Model, Scaler, and Data ---
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("Model and Scaler loaded successfully.")
except FileNotFoundError as e:
    print(f"Error: Model or scaler file not found: {e}")
    exit()

try:
    df_full = pd.read_csv(DATA_PATH)
    print(f"Dataset '{DATA_PATH}' loaded. Rows: {len(df_full)}")
    if 'Unnamed: 0' in df_full.columns:
        df_full = df_full.drop('Unnamed: 0', axis=1)

    missing_features = [f for f in FEATURES if f not in df_full.columns]
    if missing_features:
        raise ValueError(f"Missing features in dataset: {missing_features}")

    required_cols = ['track_id', 'track_name', 'artists', 'album_name', 'track_genre', 'popularity']
    missing_required = [col for col in required_cols if col not in df_full.columns]
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    df_full[FEATURES] = df_full[FEATURES].fillna(df_full[FEATURES].median())
    music_features_scaled = scaler.transform(df_full[FEATURES])
    df_full['predicted_mood'] = model.predict(music_features_scaled)
    
    # Map model predictions to AVAILABLE_MOODS if necessary
    mood_mapping = {
        # Add mappings if model labels differ, e.g., 'positive': 'Happy'
    }
    df_full['predicted_mood'] = df_full['predicted_mood'].map(mood_mapping).fillna(df_full['predicted_mood'])
    print("Mood prediction completed.")
    print("Predicted mood distribution:")
    print(df_full['predicted_mood'].value_counts())

    all_unique_songs = df_full.drop_duplicates(subset=['track_id'], keep='first')
    all_unique_songs_sorted = all_unique_songs.sort_values(by='popularity', ascending=False)
    print("Unique dataset sorted by popularity.")

except FileNotFoundError as e:
    print(f"Error: Dataset not found at '{DATA_PATH}': {e}")
    exit()
except ValueError as ve:
    print(f"Error processing dataset: {ve}")
    exit()
except Exception as e:
    print(f"Unexpected error loading/processing data: {e}")
    exit()

# --- Recommendation Functions ---
def get_recommendations_slice(mood_input, offset=0, limit=10):
    filtered_songs = all_unique_songs_sorted[all_unique_songs_sorted['predicted_mood'] == mood_input]
    total_matching_songs = len(filtered_songs)
    print(f"Debug: Mood '{mood_input}', Total songs: {total_matching_songs}, Offset: {offset}, Limit: {limit}")
    if filtered_songs.empty:
        print(f"Debug: No songs for mood '{mood_input}', falling back to popular songs")
        filtered_songs = all_unique_songs_sorted
        total_matching_songs = len(filtered_songs)
    recommendations_slice = filtered_songs.iloc[offset : offset + limit]
    required_display_cols = ['track_name', 'artists', 'album_name', 'track_genre', 'popularity', 'track_id']
    missing_display_cols = [col for col in required_display_cols if col not in recommendations_slice.columns]
    if missing_display_cols:
        print(f"Warning: Missing display columns during slicing: {missing_display_cols}")
        existing_cols = [col for col in required_display_cols if col in recommendations_slice.columns]
        return recommendations_slice[existing_cols], total_matching_songs
    return recommendations_slice[required_display_cols], total_matching_songs

def search_songs(query, offset=0, limit=10):
    """Search for songs whose titles start with the query."""
    query = query.strip().lower()
    filtered_songs = all_unique_songs_sorted[
        all_unique_songs_sorted['track_name'].str.lower().str.startswith(query, na=False)
    ]
    total_matching_songs = len(filtered_songs)
    print(f"Debug: Search query '{query}', Total songs: {total_matching_songs}, Offset: {offset}, Limit: {limit}")
    return filtered_songs.iloc[offset : offset + limit], total_matching_songs

def filter_by_letter(letter, offset=0, limit=10):
    """Search for songs whose titles start with a single letter."""
    letter = letter.strip().lower()
    filtered_songs = all_unique_songs_sorted[
        all_unique_songs_sorted['track_name'].str.lower().str.startswith(letter, na=False)
    ]
    total_matching_songs = len(filtered_songs)
    print(f"Debug: Letter '{letter}', Total songs: {total_matching_songs}, Offset: {offset}, Limit: {limit}")
    if filtered_songs.empty:
        print(f"Debug: No songs start with '{letter}', falling back to popular songs")
        filtered_songs = all_unique_songs_sorted
        total_matching_songs = len(filtered_songs)
    letter_slice = filtered_songs.iloc[offset : offset + limit]
    required_display_cols = ['track_name', 'artists', 'album_name', 'track_genre', 'popularity', 'track_id']
    return letter_slice[required_display_cols], total_matching_songs

# --- Flask Routes ---
@app.route('/')
def index():
    return render_template('index.html', moods=AVAILABLE_MOODS)

@app.route('/recommend', methods=['POST'])
def recommend():
    selected_mood = request.form.get('mood')
    recommendations_result = None
    error_message = None
    total_songs_for_mood = 0

    if not selected_mood:
        error_message = "Please select a mood."
    elif selected_mood not in AVAILABLE_MOODS:
        error_message = f"Invalid mood: '{selected_mood}'."
    else:
        result_df, total_songs_for_mood = get_recommendations_slice(
            selected_mood, offset=0, limit=INITIAL_LOAD_COUNT
        )
        if not result_df.empty:
            recommendations_result = result_df
        else:
            error_message = f"No songs found for mood '{selected_mood}'. Try another mood or search for a song."

    show_more_possible = total_songs_for_mood > INITIAL_LOAD_COUNT
    return render_template('recommendations.html',
                          selected_mood=selected_mood,
                          recommendations=recommendations_result,
                          error=error_message,
                          show_more=show_more_possible,
                          initial_count=INITIAL_LOAD_COUNT,
                          load_more_count=LOAD_MORE_COUNT)

@app.route('/api/recommendations')
def api_recommendations():
    mood = request.args.get('mood')
    try:
        offset = int(request.args.get('offset', 0))
    except ValueError:
        return jsonify({"error": "Invalid offset value"}), 400

    if not mood or mood not in AVAILABLE_MOODS:
        return jsonify({"error": "Invalid or unknown mood"}), 400

    result_df, total_songs_for_mood = get_recommendations_slice(mood, offset=offset, limit=LOAD_MORE_COUNT)
    recommendations_json = result_df.to_dict(orient='records')
    has_more = (offset + len(recommendations_json)) < total_songs_for_mood
    return jsonify({
        "recommendations": recommendations_json,
        "has_more": has_more
    })

@app.route('/api/search')
def api_search():
    query = request.args.get('query', '')
    try:
        offset = int(request.args.get('offset', 0))
    except ValueError:
        return jsonify({"error": "Invalid offset value"}), 400

    if not query:
        return jsonify({"error": "Query is required"}), 400

    result_df, total_songs = search_songs(query, offset=offset, limit=LOAD_MORE_COUNT)
    if result_df.empty:
        return jsonify({"error": f"No songs found starting with '{query}'"})
    search_json = result_df.to_dict(orient='records')
    has_more = (offset + len(search_json)) < total_songs
    return jsonify({
        "recommendations": search_json,
        "has_more": has_more
    })

@app.route('/api/filter_by_letter')
def api_filter_by_letter():
    letter = request.args.get('letter', '')
    try:
        offset = int(request.args.get('offset', 0))
    except ValueError:
        return jsonify({"error": "Invalid offset value"}), 400

    if not letter or len(letter) != 1 or not letter.isalpha():
        return jsonify({"error": "Letter must be a single alphabet character"}), 400

    result_df, total_songs = filter_by_letter(letter, offset=offset, limit=LOAD_MORE_COUNT)
    filter_json = result_df.to_dict(orient='records')
    has_more = (offset + len(filter_json)) < total_songs
    return jsonify({
        "recommendations": filter_json,
        "has_more": has_more
    })

# --- Run Application ---
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)