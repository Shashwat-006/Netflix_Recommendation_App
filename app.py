import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------------
# Page Config
# -----------------------------------
st.set_page_config(
    page_title="Netflix Recommender",
    page_icon="🎬",
    layout="centered"
)

# -----------------------------------
# Custom CSS
# -----------------------------------
st.markdown("""
<style>
.movie-card {
    padding: 1rem;
    margin-bottom: 1rem;
    border-radius: 12px;
    background: #1c1c1c;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.3);
}
.movie-title {
    font-size: 20px;
    font-weight: 600;
}
.movie-meta {
    color: #bbbbbb;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------
# Title Section
# -----------------------------------
st.title("🎬 Netflix Recommendation System")
st.caption("Personalized content-based movie & TV show recommendations")

# -----------------------------------
# Sidebar Controls
# -----------------------------------
st.sidebar.header("⚙️ Recommendation Settings")

num_recommendations = st.sidebar.slider(
    "Number of recommendations",
    min_value=3,
    max_value=10,
    value=5
)

# -----------------------------------
# Load Data
# -----------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("netflix_dataset.csv")

df = load_data()

# -----------------------------------
# Preprocessing
# -----------------------------------
df['combined_features'] = (
    df['type'].fillna('') + " " +
    df['listed_in'].fillna('') + " " +
    df['description'].fillna('')
)

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['combined_features'])

cosine_sim = cosine_similarity(tfidf_matrix)

# -----------------------------------
# Recommendation Function
# -----------------------------------
def recommend(title, n):
    idx = df[df['title'] == title].index[0]
    scores = list(enumerate(cosine_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    scores = scores[1:n + 1]

    indices = [i[0] for i in scores]
    return df.iloc[indices]

# -----------------------------------
# Main UI
# -----------------------------------
st.subheader("🔍 Select a Movie or TV Show")

selected_title = st.selectbox(
    "",
    sorted(df['title'].dropna().unique())
)

st.markdown("---")

if st.button("✨ Get Recommendations"):
    results = recommend(selected_title, num_recommendations)

    st.subheader("🎯 Recommended for You")

    for _, row in results.iterrows():
        st.markdown(f"""
        <div class="movie-card">
            <div class="movie-title">🎥 {row['title']}</div>
            <div class="movie-meta">
                {row['type']} • {row['listed_in']}
            </div>
            <br>
            <div class="movie-meta">
                {row['description'][:250]}...
            </div>
        </div>
        """, unsafe_allow_html=True)
st.caption("Built by Shashwat using Streamlit")