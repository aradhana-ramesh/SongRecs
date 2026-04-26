from pathlib import Path
from difflib import get_close_matches

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
            align-items: start;
        }
        .search-wrap {
            position: relative;
        }
        input[type="text"] {
            width: 100%;
            padding: 14px 16px;
            border: 1px solid var(--border);
            border-radius: 8px;
            font-size: 1rem;
            background: #fff;
        }
        .suggestions {
            position: absolute;
            top: calc(100% + 6px);
            left: 0;
            right: 0;
            background: #fff;
            border: 1px solid var(--border);
            border-radius: 8px;
            box-shadow: 0 16px 32px rgba(31, 41, 51, 0.12);
            max-height: 220px;
            overflow-y: auto;
            display: none;
            z-index: 10;
        }
        .suggestions.visible {
            display: block;
        }
        .suggestion-item {
            width: 100%;
            border: 0;
            border-bottom: 1px solid #f1e7dc;
            background: #fff;
            color: var(--ink);
            text-align: left;
            padding: 12px 14px;
            font-size: 0.98rem;
            cursor: pointer;
        }
        .suggestion-item:last-child {
            border-bottom: 0;
        }
        .suggestion-item:hover,
        .suggestion-item.active {
            background: #fff3e8;
            color: var(--accent-dark);
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
            <div class="search-wrap">
                <input
                    id="song_name"
                    type="text"
                    name="song_name"
                    placeholder="Try: Aperture"
                    value="{{ query }}"
                    autocomplete="off"
                    required
                >
                <div id="suggestions" class="suggestions"></div>
            </div>
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
    <script>
        const songOptions = {{ song_options|tojson }};
        const input = document.getElementById("song_name");
        const suggestions = document.getElementById("suggestions");
        let activeIndex = -1;
        let visibleMatches = [];

        function closeSuggestions() {
            suggestions.classList.remove("visible");
            suggestions.innerHTML = "";
            activeIndex = -1;
            visibleMatches = [];
        }

        function selectSuggestion(song) {
            input.value = song.track;
            closeSuggestions();
            input.focus();
        }

        function renderSuggestions(matches) {
            visibleMatches = matches;
            activeIndex = -1;

            if (!matches.length) {
                closeSuggestions();
                return;
            }

            suggestions.innerHTML = matches.map((song) =>
                `<button type="button" class="suggestion-item">${song.track} - ${song.artist}</button>`
            ).join("");

            suggestions.classList.add("visible");

            suggestions.querySelectorAll(".suggestion-item").forEach((item, index) => {
                item.addEventListener("mousedown", (event) => {
                    event.preventDefault();
                    selectSuggestion(matches[index]);
                });
            });
        }

        function updateActiveItem() {
            const items = suggestions.querySelectorAll(".suggestion-item");
            items.forEach((item, index) => {
                item.classList.toggle("active", index === activeIndex);
            });

            if (activeIndex >= 0 && items[activeIndex]) {
                items[activeIndex].scrollIntoView({ block: "nearest" });
            }
        }

        input.addEventListener("input", () => {
            const query = input.value.trim().toLowerCase();
            if (!query) {
                closeSuggestions();
                return;
            }

            const matches = songOptions
                .filter((song) =>
                    song.track.toLowerCase().includes(query) ||
                    song.artist.toLowerCase().includes(query) ||
                    `${song.track} - ${song.artist}`.toLowerCase().includes(query)
                )
                .slice(0, 12);

            renderSuggestions(matches);
        });

        input.addEventListener("focus", () => {
            if (input.value.trim()) {
                const query = input.value.trim().toLowerCase();
                const matches = songOptions
                    .filter((song) =>
                        song.track.toLowerCase().includes(query) ||
                        song.artist.toLowerCase().includes(query) ||
                        `${song.track} - ${song.artist}`.toLowerCase().includes(query)
                    )
                    .slice(0, 12);
                renderSuggestions(matches);
            }
        });

        input.addEventListener("keydown", (event) => {
            if (!visibleMatches.length) {
                return;
            }

            if (event.key === "ArrowDown") {
                event.preventDefault();
                activeIndex = (activeIndex + 1) % visibleMatches.length;
                updateActiveItem();
            } else if (event.key === "ArrowUp") {
                event.preventDefault();
                activeIndex = (activeIndex - 1 + visibleMatches.length) % visibleMatches.length;
                updateActiveItem();
            } else if (event.key === "Enter" && activeIndex >= 0) {
                event.preventDefault();
                selectSuggestion(visibleMatches[activeIndex]);
            } else if (event.key === "Escape") {
                closeSuggestions();
            }
        });

        document.addEventListener("click", (event) => {
            if (!event.target.closest(".search-wrap")) {
                closeSuggestions();
            }
        });
    </script>
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

track_names = df["track"].dropna().tolist()
track_names_lower = [track.lower() for track in track_names]
song_options = (
    df[["artist", "track"]]
    .dropna()
    .drop_duplicates()
    .to_dict(orient="records")
)


def resolve_track_name(track_name):
    matches = df[(df["track"].str.lower() == track_name.lower())]
    if not matches.empty:
        return track_name, None

    partial_matches = df[df["track"].str.lower().str.contains(track_name.lower(), na=False)]
    if not partial_matches.empty:
        matched_track = partial_matches.iloc[0]["track"]
        return matched_track, f'No exact match found. Showing results for "{matched_track}".'

    close_matches = get_close_matches(track_name.lower(), track_names_lower, n=1, cutoff=0.6)
    if close_matches:
        matched_lower = close_matches[0]
        matched_track = track_names[track_names_lower.index(matched_lower)]
        return matched_track, f'No exact match found. Showing results for "{matched_track}".'

    return None, None


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
            resolved_track, info = resolve_track_name(query)
            if resolved_track is None:
                error = "Track name not found. Try again."
            else:
                recommended_songs, similarity_score = recommend(resolved_track)
                selected_track = resolved_track
                results = recommended_songs[["artist", "track"]].head(5).to_dict(orient="records")

    return render_template_string(
        HTML_TEMPLATE,
        query=query,
        results=results,
        error=error,
        info=info,
        similarity_score=similarity_score,
        selected_track=selected_track,
        song_titles=track_names,
        song_options=song_options,
    )


if __name__ == "__main__":
    app.run(debug=True)
