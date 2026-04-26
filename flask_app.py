from pathlib import Path

import pandas as pd
from flask import Flask, render_template_string, request
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "preprocessed_songs.csv"


HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Song Recommender</title>
    <style>
        :root {
            --bg: #f5efe6;
            --panel: #fffaf4;
            --ink: #1f2933;
            --muted: #52606d;
            --accent: #c2410c;
            --accent-dark: #9a3412;
            --border: #e7d8c9;
        }
        * { box-sizing: border-box; }
        body {
            margin: 0;
            font-family: Georgia, "Times New Roman", serif;
            background:
                radial-gradient(circle at top left, #ffe3c2 0, transparent 28%),
                radial-gradient(circle at bottom right, #f7c59f 0, transparent 25%),
                var(--bg);
            color: var(--ink);
            min-height: 100vh;
            display: grid;
            place-items: center;
            padding: 24px;
        }
        .shell {
            width: min(760px, 100%);
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 28px;
            box-shadow: 0 18px 40px rgba(31, 41, 51, 0.08);
        }
        h1 {
            margin: 0 0 8px;
            font-size: clamp(2rem, 4vw, 3rem);
            line-height: 1.05;
            font-weight: 700;
        }
        p {
            margin: 0 0 20px;
            color: var(--muted);
            font-size: 1.05rem;
        }
        form {
            display: grid;
            grid-template-columns: 1fr auto;
            gap: 12px;
            margin-bottom: 24px;
        }
        input[type="text"] {
            width: 100%;
            padding: 14px 16px;
            border: 1px solid var(--border);
            border-radius: 8px;
            font-size: 1rem;
            background: #fff;
        }
        button {
            border: 0;
            border-radius: 8px;
            background: var(--accent);
            color: white;
            padding: 14px 18px;
            font-size: 1rem;
            cursor: pointer;
        }
        button:hover {
            background: var(--accent-dark);
        }
        .message {
            padding: 14px 16px;
            border-radius: 8px;
            margin-bottom: 18px;
        }
        .message.error {
            background: #fff1f2;
            color: #9f1239;
            border: 1px solid #fecdd3;
        }
        .message.info {
            background: #fff7ed;
            color: #9a3412;
            border: 1px solid #fed7aa;
        }
        .results h2 {
            margin: 0 0 8px;
            font-size: 1.35rem;
        }
        .results-meta {
            margin-bottom: 16px;
            color: var(--muted);
        }
        ul {
            list-style: none;
            margin: 0;
            padding: 0;
            display: grid;
            gap: 12px;
        }
        li {
            background: #fff;
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 14px 16px;
        }
        .track {
            font-size: 1.05rem;
            font-weight: 700;
            margin-bottom: 4px;
        }
        .artist, .score {
            color: var(--muted);
        }
        @media (max-width: 640px) {
            form {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <main class="shell">
        <h1>Song Recommender</h1>
        <p>Enter a song title and get the five most similar songs from your KNN model.</p>

        <form method="post">
            <input
                type="text"
                name="song_name"
                placeholder="Try: Aperture"
                value="{{ query }}"
                autocomplete="off"
                required
            >
            <button type="submit">Search</button>
        </form>

        {% if error %}
            <div class="message error">{{ error }}</div>
        {% endif %}

        {% if info %}
            <div class="message info">{{ info }}</div>
        {% endif %}

        {% if results %}
            <section class="results">
                <h2>Recommendations for "{{ selected_track }}"</h2>
                <div class="results-meta">
                    Similarity Score: {{ "%.3f"|format(similarity_score) }}
                </div>
                <ul>
                    {% for song in results %}
                        <li>
                            <div class="track">{{ song.track }}</div>
                            <div class="artist">{{ song.artist }}</div>
                        </li>
                    {% endfor %}
                </ul>
            </section>
        {% endif %}
    </main>
</body>
</html>
"""


df = pd.read_csv(DATA_PATH)
genre_cols = df.columns[7:]
df = df[df[genre_cols].sum(axis=1) > 0]
df = df.reset_index(drop=True)

dfclean = df.drop(columns=["Unnamed: 0", "artist", "track", "tags"])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(dfclean)

model = NearestNeighbors(n_neighbors=10, metric="cosine")
model.fit(X_scaled)


def recommend(track_name, k=10):
    index = get_index(track_name)
    if index is None:
        return None, 0
    recommended_songs, indices = get_reccs(index)
    score = get_similarity_score(index, indices)
    return recommended_songs, score


def get_index(track_name):
    matches = df[(df["track"].str.lower() == track_name.lower())]
    if matches.empty:
        print("Track name not found. Try again.")
        return None
    return matches.index[0]


def get_reccs(song_index):
    distances, indices = model.kneighbors(X_scaled[song_index].reshape(1, -1))
    indices = indices[0][1:]
    recommended_songs = df.iloc[indices]
    print(recommended_songs[["artist", "track"]])
    return recommended_songs, indices


def get_similarity_score(song_index, indices):
    original = df.iloc[song_index][genre_cols]
    recs = df.iloc[indices][genre_cols]

    active_genres = original[original == 1].index
    matches = recs[active_genres].sum(axis=1)
    consistency_score = matches.mean() / len(active_genres)

    print("Similarity Score:", consistency_score)
    return consistency_score


@app.route("/", methods=["GET", "POST"])
def index():
    results = None
    error = None
    info = None
    similarity_score = 0.0
    selected_track = ""
    query = ""

    if request.method == "POST":
        query = request.form.get("song_name", "").strip()
        if not query:
            error = "Please enter a song title."
        else:
            recommended_songs, similarity_score = recommend(query)
            if recommended_songs is None:
                error = "Track name not found. Try again."
            else:
                selected_track = query
                results = recommended_songs[["artist", "track"]].head(5).to_dict(orient="records")

    return render_template_string(
        HTML_TEMPLATE,
        query=query,
        results=results,
        error=error,
        info=info,
        similarity_score=similarity_score,
        selected_track=selected_track,
    )


if __name__ == "__main__":
    app.run(debug=True)
