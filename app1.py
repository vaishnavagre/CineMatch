import streamlit as st
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="CineMatch | Movie Recommendation",
    page_icon="🎞️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS — design tokens grounded in the film-strip /
# ticket-stub / marquee world instead of generic dark cards
# =========================================================

st.markdown("""
<style>

    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600&display=swap');

    :root {
        --bg: #0b0b0d;
        --surface: #17151b;
        --surface-alt: #1d1a22;
        --border: #2a2733;
        --gold: #e8b94a;
        --gold-dim: #a9843a;
        --curtain: #8c2f26;
        --text: #f3efe8;
        --text-muted: #948f9c;
        --serif: 'Fraunces', Georgia, serif;
        --sans: 'Inter', -apple-system, sans-serif;
    }

    .stApp { background-color: var(--bg); }
    html, body, [class*="css"] { font-family: var(--sans); }

    /* ---- reduced motion respect ---- */
    @media (prefers-reduced-motion: reduce) {
        * { transition: none !important; animation: none !important; }
    }

    /* ---- Sidebar ---- */
    section[data-testid="stSidebar"] {
        background-color: var(--surface);
        border-right: 1px solid var(--border);
    }
    section[data-testid="stSidebar"] * { color: var(--text) !important; }
    section[data-testid="stSidebar"] .sb-eyebrow {
        font-family: var(--serif);
        font-size: 22px;
        font-weight: 600;
        color: var(--gold) !important;
        margin-bottom: 2px;
    }
    section[data-testid="stSidebar"] .sb-tag {
        color: var(--text-muted) !important;
        font-size: 13.5px;
        line-height: 1.5;
    }
    .sb-step {
        display: flex;
        gap: 10px;
        padding: 6px 0;
        border-left: 2px solid var(--border);
        padding-left: 12px;
        margin-left: 2px;
    }
    .sb-step-num {
        font-family: var(--serif);
        color: var(--gold) !important;
        font-size: 14px;
        min-width: 16px;
    }
    .sb-step-text { font-size: 13.5px; color: var(--text) !important; }
    .sb-method {
        background: var(--surface-alt);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 10px 12px;
        margin-bottom: 8px;
    }
    .sb-method-name {
        font-weight: 600;
        font-size: 13.5px;
        color: var(--gold) !important;
    }
    .sb-method-desc {
        font-size: 12.5px;
        color: var(--text-muted) !important;
        margin-top: 2px;
    }

    /* ---- Header: marquee ---- */
    .marquee-wrap {
        border-bottom: 1px solid var(--border);
        padding-bottom: 22px;
        margin-bottom: 28px;
    }
    .marquee-eyebrow {
        font-size: 13px;
        color: var(--gold-dim);
        letter-spacing: 0.02em;
        margin-bottom: 6px;
    }
    .main-title {
        font-family: var(--serif);
        font-size: 52px;
        font-weight: 600;
        color: var(--text);
        line-height: 1.05;
        margin: 0;
    }
    .subtitle {
        color: var(--text-muted);
        font-size: 16.5px;
        margin-top: 10px;
        max-width: 620px;
        line-height: 1.5;
    }

    /* ---- Section labels ---- */
    .section-label {
        font-family: var(--serif);
        font-size: 22px;
        font-weight: 600;
        color: var(--text);
        margin-bottom: 4px;
    }

    /* ---- Ticket stub (selected movie) ---- */
    .ticket {
        position: relative;
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 4px;
        margin: 18px 0 26px 0;
        display: flex;
        overflow: hidden;
    }
    .ticket-main {
        flex: 1;
        padding: 22px 26px;
    }
    .ticket-label {
        color: var(--text-muted);
        font-size: 12.5px;
        margin-bottom: 6px;
    }
    .ticket-movie {
        font-family: var(--serif);
        font-size: 30px;
        font-weight: 600;
        color: var(--text);
    }
    .ticket-stub-side {
        width: 90px;
        border-left: 2px dashed var(--border);
        background: var(--surface-alt);
        display: flex;
        align-items: center;
        justify-content: center;
        position: relative;
    }
    .ticket-stub-side::before,
    .ticket-stub-side::after {
        content: "";
        position: absolute;
        width: 18px;
        height: 18px;
        background: var(--bg);
        border-radius: 50%;
        left: -10px;
    }
    .ticket-stub-side::before { top: -9px; }
    .ticket-stub-side::after { bottom: -9px; }
    .ticket-stub-icon {
        font-size: 26px;
        color: var(--gold);
        writing-mode: vertical-rl;
        letter-spacing: 3px;
        font-family: var(--serif);
        font-size: 13px;
        color: var(--gold-dim);
    }

    /* ---- Buttons ---- */
    .stButton > button {
        width: 100%;
        border-radius: 6px;
        height: 46px;
        font-weight: 600;
        font-size: 15px;
        background: var(--gold);
        color: #17140c;
        border: none;
        transition: background 0.15s ease, transform 0.1s ease;
    }
    .stButton > button:hover {
        background: #f2c968;
        transform: translateY(-1px);
    }
    .stButton > button:active { transform: translateY(0); }

    /* ---- Film-frame recommendation rows ---- */
    .frame {
        position: relative;
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 4px;
        margin-bottom: 12px;
        display: flex;
        overflow: hidden;
    }
    .frame-sprockets {
        width: 22px;
        flex-shrink: 0;
        background-image: repeating-radial-gradient(
            circle at 11px 14px,
            var(--bg) 0 4px,
            transparent 4px 24px
        );
        background-color: var(--surface-alt);
        border-right: 1px solid var(--border);
    }
    .frame-body {
        flex: 1;
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 16px 22px;
        gap: 18px;
    }
    .frame-rank {
        font-family: var(--serif);
        font-size: 34px;
        font-weight: 600;
        color: var(--border);
        min-width: 40px;
    }
    .frame-title-wrap { flex: 1; min-width: 0; }
    .frame-title {
        font-family: var(--serif);
        font-size: 19px;
        font-weight: 600;
        color: var(--text);
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .frame-score-wrap {
        min-width: 160px;
        text-align: right;
    }
    .frame-score-label {
        font-size: 12px;
        color: var(--text-muted);
        margin-bottom: 5px;
    }
    .frame-bar-track {
        height: 5px;
        border-radius: 3px;
        background: var(--border);
        overflow: hidden;
    }
    .frame-bar-fill {
        height: 100%;
        background: var(--gold);
        border-radius: 3px;
    }

    /* ---- Metrics ---- */
    [data-testid="stMetricValue"] { color: var(--gold) !important; font-family: var(--serif); }
    [data-testid="stMetricLabel"] { color: var(--text-muted) !important; }

    /* ---- Footer ---- */
    .footer-row {
        text-align: center;
        color: var(--text-muted);
        font-size: 13px;
        padding-top: 6px;
    }

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    with open("movies.pkl", "rb") as file:
        movies = pickle.load(file)

    return movies


movies = load_data()


# =========================================================
# CREATE SIMILARITY MATRIX
# =========================================================

@st.cache_resource
def create_similarity(movies):

    cv = CountVectorizer(
        max_features=5000,
        stop_words="english"
    )

    vectors = cv.fit_transform(
        movies["tags"].fillna("")
    ).toarray()

    similarity = cosine_similarity(vectors)

    return similarity


similarity = create_similarity(movies)


# =========================================================
# RECOMMENDATION FUNCTION
# =========================================================

def recommend(movie):

    movie_index = movies[
        movies["title"] == movie
    ].index[0]

    distances = similarity[movie_index]

    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    recommendations = []

    for index, score in movie_list:

        movie_name = movies.iloc[index]["title"]

        recommendations.append(
            (movie_name, score)
        )

    return recommendations


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown('<div class="sb-eyebrow">CineMatch</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sb-tag">A content-based recommendation engine '
        'that finds films by what they\'re actually about.</div>',
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    steps = [
        "Pick a movie you like",
        "Its tags are parsed into a vector",
        "Every other film is compared against it",
        "Results are ranked by closeness",
        "Top 5 matches are shown",
    ]
    for i, step in enumerate(steps, start=1):
        st.markdown(
            f'<div class="sb-step">'
            f'<div class="sb-step-num">{i}</div>'
            f'<div class="sb-step-text">{step}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        '<div class="sb-method">'
        '<div class="sb-method-name">CountVectorizer</div>'
        '<div class="sb-method-desc">Turns each film\'s tags into a numerical vector.</div>'
        '</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="sb-method">'
        '<div class="sb-method-name">Cosine Similarity</div>'
        '<div class="sb-method-desc">Measures how closely two vectors point in the same direction.</div>'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sb-tag" style="margin-top:18px; font-size:12px;">'
        'Machine learning project · content-based filtering</div>',
        unsafe_allow_html=True
    )


# =========================================================
# MAIN HEADER
# =========================================================

st.markdown(
    """
    <div class="marquee-wrap">
        <div class="marquee-eyebrow">Now recommending</div>
        <div class="main-title">CineMatch</div>
        <div class="subtitle">
            Pick a film you already love — CineMatch reads its tags and
            finds the five closest matches in the catalog.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# MOVIE SELECTION
# =========================================================

st.markdown('<div class="section-label">Find your next movie</div>', unsafe_allow_html=True)

movie_list = movies["title"].dropna().tolist()

selected_movie = st.selectbox(
    "Choose a movie you like",
    movie_list,
    index=0,
    label_visibility="collapsed"
)


# =========================================================
# SELECTED MOVIE — ticket stub
# =========================================================

st.markdown(
    f"""
    <div class="ticket">
        <div class="ticket-main">
            <div class="ticket-label">Selected</div>
            <div class="ticket-movie">{selected_movie}</div>
        </div>
        <div class="ticket-stub-side">
            <div class="ticket-stub-icon">ADMIT ONE</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# BUTTON
# =========================================================

col1, col2, col3 = st.columns([1, 1, 1])

with col2:

    recommend_button = st.button("Show recommendations")


# =========================================================
# RECOMMENDATIONS — film-frame strips
# =========================================================

if recommend_button:

    recommendations = recommend(selected_movie)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        f'<div class="section-label">Similar to {selected_movie}</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="subtitle" style="margin-bottom:16px;">'
        'Ranked by how closely each film\'s tags match your pick.</div>',
        unsafe_allow_html=True
    )

    for i, (movie, score) in enumerate(recommendations, start=1):

        percentage = round(score * 100, 2)

        st.markdown(
            f"""
            <div class="frame">
                <div class="frame-sprockets"></div>
                <div class="frame-body">
                    <div class="frame-rank">{i:02d}</div>
                    <div class="frame-title-wrap">
                        <div class="frame-title">{movie}</div>
                    </div>
                    <div class="frame-score-wrap">
                        <div class="frame-score-label">{percentage}% match</div>
                        <div class="frame-bar-track">
                            <div class="frame-bar-fill" style="width:{percentage}%;"></div>
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


# =========================================================
# PROJECT INFORMATION
# =========================================================

st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<hr style="border-color: #2a2733;">', unsafe_allow_html=True)
st.markdown('<div class="section-label">About this system</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Movies in dataset", len(movies))

with col2:
    st.metric("Recommendations shown", "5")

with col3:
    st.metric("Method", "Cosine similarity")


# =========================================================
# FOOTER
# =========================================================

st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<hr style="border-color: #2a2733;">', unsafe_allow_html=True)
st.markdown(
    '<div class="footer-row">CineMatch — content-based movie recommendation system</div>',
    unsafe_allow_html=True
)