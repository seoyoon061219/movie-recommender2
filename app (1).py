"""
🎬 TMDb 영화 추천 시스템 - Streamlit 앱
실행: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────
# 페이지 설정
# ─────────────────────────────────────────
st.set_page_config(
    page_title="🎬 영화 추천 시스템",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────
# 커스텀 CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500&display=swap');

/* 전체 배경 */
.stApp {
    background: #0a0a0f;
    color: #e8e4dc;
}

/* 사이드바 */
section[data-testid="stSidebar"] {
    background: #111118;
    border-right: 1px solid #2a2a3a;
}

/* 헤더 */
.hero-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 4rem;
    letter-spacing: 0.12em;
    color: #f5c518;
    line-height: 1;
    margin-bottom: 0.2rem;
}
.hero-sub {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.95rem;
    color: #666;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 2rem;
}

/* 영화 카드 */
.movie-card {
    background: #13131c;
    border: 1px solid #222230;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.8rem;
    transition: border-color 0.2s;
    font-family: 'DM Sans', sans-serif;
}
.movie-card:hover {
    border-color: #f5c518;
}
.movie-rank {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2rem;
    color: #2a2a3a;
    line-height: 1;
    float: left;
    margin-right: 1rem;
}
.movie-title {
    font-size: 1.05rem;
    font-weight: 500;
    color: #f0ece2;
    margin-bottom: 0.25rem;
}
.movie-meta {
    font-size: 0.8rem;
    color: #555;
}
.movie-rating {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.3rem;
    color: #f5c518;
}
.genre-tag {
    display: inline-block;
    background: #1e1e2e;
    border: 1px solid #333;
    border-radius: 4px;
    padding: 0.1rem 0.5rem;
    font-size: 0.72rem;
    color: #888;
    margin-right: 0.3rem;
    margin-top: 0.3rem;
}
.score-bar-bg {
    background: #1e1e2e;
    border-radius: 4px;
    height: 4px;
    margin-top: 0.5rem;
}
.score-bar-fill {
    background: linear-gradient(90deg, #f5c518, #e0a800);
    border-radius: 4px;
    height: 4px;
}
.selected-movie-box {
    background: linear-gradient(135deg, #1a1a28, #13131c);
    border: 1px solid #f5c51855;
    border-radius: 12px;
    padding: 1.4rem;
    margin-bottom: 2rem;
    font-family: 'DM Sans', sans-serif;
}
.divider {
    border: none;
    border-top: 1px solid #1e1e2e;
    margin: 1.5rem 0;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# 데이터 & 모델 (캐시)
# ─────────────────────────────────────────

def parse_names(obj, top=5):
    try:
        return [d['name'].replace(' ', '') for d in ast.literal_eval(obj)[:top]]
    except:
        return []

def get_director(crew_str):
    try:
        for c in ast.literal_eval(crew_str):
            if c.get('job') == 'Director':
                return c['name']
    except:
        pass
    return '알 수 없음'

def make_soup(row):
    overview_words = row['overview'].split() if isinstance(row['overview'], str) else []
    parts = (
        row['genres_list']
        + row['keywords_list']
        + row['cast_list']
        + [row['director'].replace(' ', '')] * 2
        + overview_words
    )
    return ' '.join(parts)

def compute_weighted_score(df, percentile=0.70):
    m = df['vote_count'].quantile(percentile)
    C = df['vote_average'].mean()
    return df.apply(lambda r: (r['vote_count']/(r['vote_count']+m))*r['vote_average']
                               + (m/(r['vote_count']+m))*C, axis=1)

@st.cache_resource(show_spinner="📂 데이터 로드 & 모델 구축 중... (최초 1회)")
def load_and_build(movies_path, credits_path):
    movies  = pd.read_csv(movies_path)
    credits = pd.read_csv(credits_path)
    credits.rename(columns={'movie_id': 'id'}, inplace=True)
    df = movies.merge(credits, on='id').dropna(subset=['overview']).reset_index(drop=True)

    df['genres_list']   = df['genres'].apply(parse_names)
    df['keywords_list'] = df['keywords'].apply(lambda x: parse_names(x, top=10))
    df['cast_list']     = df['cast'].apply(lambda x: parse_names(x, top=3))
    df['director']      = df['crew'].apply(get_director)
    df['soup']          = df.apply(make_soup, axis=1)

    cv         = CountVectorizer(stop_words='english')
    matrix     = cv.fit_transform(df['soup'])
    cosine_sim = cosine_similarity(matrix, matrix)
    indices    = pd.Series(df.index, index=df['title']).drop_duplicates()

    return df, cosine_sim, indices


def recommend(title, df, cosine_sim, indices, top_n, content_w, score_w):
    if title not in indices:
        return None
    idx = indices[title]
    sim_scores = sorted(enumerate(cosine_sim[idx]), key=lambda x: x[1], reverse=True)[1:top_n*5]
    movie_indices = [i[0] for i in sim_scores]
    sim_values    = [i[1] for i in sim_scores]

    cand = df.iloc[movie_indices].copy()
    cand['content_score'] = sim_values

    ws = compute_weighted_score(cand)
    cand['weighted_score'] = ws
    ws_min, ws_max = ws.min(), ws.max()
    cand['ws_norm'] = (ws - ws_min) / (ws_max - ws_min) if ws_max > ws_min else 0.5

    cand['hybrid_score'] = content_w * cand['content_score'] + score_w * cand['ws_norm']

    result = (cand.sort_values('hybrid_score', ascending=False).head(top_n)
              [['title','genres_list','director','vote_average','vote_count','hybrid_score','overview']]
              .rename(columns={'genres_list':'genres',
                               'vote_average':'rating','vote_count':'votes'}))
    return result.reset_index(drop=True)


# ─────────────────────────────────────────
# 사이드바
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ 설정")
    st.markdown("---")

    movies_path  = st.text_input("movies CSV 경로", value="tmdb_5000_movies.csv")
    credits_path = st.text_input("credits CSV 경로", value="tmdb_5000_credits_slim.csv")

    st.markdown("---")
    st.markdown("**추천 가중치**")
    content_w = st.slider("콘텐츠 유사도", 0.0, 1.0, 0.7, 0.05)
    score_w   = round(1 - content_w, 2)
    st.caption(f"인기도/평점 가중치: **{score_w}** (자동)")

    st.markdown("---")
    top_n = st.slider("추천 영화 수", 5, 20, 10)


# ─────────────────────────────────────────
# 메인
# ─────────────────────────────────────────
st.markdown('<div class="hero-title">MOVIE FINDER</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">TMDb 기반 콘텐츠 하이브리드 추천 시스템</div>', unsafe_allow_html=True)

# 데이터 로드
try:
    df, cosine_sim, indices = load_and_build(movies_path, credits_path)
except FileNotFoundError:
    st.error("❌ CSV 파일을 찾을 수 없어요. 사이드바에서 경로를 확인해 주세요.")
    st.stop()

# 영화 검색
movie_list = sorted(indices.index.tolist())
selected   = st.selectbox(
    "🎬 영화를 선택하세요",
    options=[""] + movie_list,
    format_func=lambda x: "검색하려면 입력하세요..." if x == "" else x,
)

if not selected:
    # 기본 화면 — 인기 영화 Top 10
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown("#### 📈 평점 높은 영화 TOP 10")
    top_movies = (df[df['vote_count'] > 1000]
                  .sort_values('vote_average', ascending=False)
                  .head(10)[['title','genres_list','director','vote_average','vote_count']]
                  .reset_index(drop=True))

    for i, row in top_movies.iterrows():
        genres_html = ''.join(f'<span class="genre-tag">{g}</span>' for g in row['genres_list'])
        st.markdown(f"""
        <div class="movie-card">
            <span class="movie-rank">#{i+1}</span>
            <div class="movie-title">{row['title']}</div>
            <div class="movie-meta">🎬 {row['director']} &nbsp;|&nbsp;
                <span class="movie-rating">★ {row['vote_average']}</span>
                &nbsp;({int(row['vote_count']):,}명)
            </div>
            <div>{genres_html}</div>
        </div>
        """, unsafe_allow_html=True)

else:
    # 선택된 영화 정보 표시
    movie_info = df[df['title'] == selected].iloc[0]
    genres_html = ''.join(f'<span class="genre-tag">{g}</span>' for g in movie_info['genres_list'])
    overview_text = movie_info['overview'] if isinstance(movie_info['overview'], str) else ''

    st.markdown(f"""
    <div class="selected-movie-box">
        <div style="font-family:'Bebas Neue',sans-serif;font-size:1.8rem;color:#f5c518;letter-spacing:0.08em;">
            {selected}
        </div>
        <div style="font-size:0.85rem;color:#666;margin:0.3rem 0 0.6rem;">
            🎬 {movie_info['director']} &nbsp;|&nbsp; ★ {movie_info['vote_average']} ({int(movie_info['vote_count']):,}명)
        </div>
        <div>{genres_html}</div>
        <div style="font-size:0.85rem;color:#888;margin-top:0.8rem;line-height:1.6;">{overview_text[:200]}{'...' if len(overview_text)>200 else ''}</div>
    </div>
    """, unsafe_allow_html=True)

    # 추천 실행
    with st.spinner("🔍 추천 중..."):
        result = recommend(selected, df, cosine_sim, indices, top_n, content_w, score_w)

    st.markdown(f"#### 🎯 '{selected}' 와(과) 비슷한 영화 TOP {top_n}")
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    if result is None or result.empty:
        st.warning("추천 결과를 찾지 못했어요.")
    else:
        max_score = result['hybrid_score'].max()
        for i, row in result.iterrows():
            genres_html = ''.join(f'<span class="genre-tag">{g}</span>' for g in row['genres'])
            bar_pct     = int((row['hybrid_score'] / max_score) * 100) if max_score > 0 else 0
            overview_short = row['overview'][:120] + '...' if isinstance(row['overview'], str) and len(row['overview']) > 120 else row.get('overview', '')

            st.markdown(f"""
            <div class="movie-card">
                <span class="movie-rank">#{i+1}</span>
                <div class="movie-title">{row['title']}</div>
                <div class="movie-meta">
                    🎬 {row['director']} &nbsp;|&nbsp;
                    <span class="movie-rating">★ {row['rating']}</span>
                    &nbsp;({int(row['votes']):,}명)
                    &nbsp;|&nbsp; 유사도 점수: <b style="color:#f5c518">{row['hybrid_score']:.4f}</b>
                </div>
                <div>{genres_html}</div>
                <div class="score-bar-bg"><div class="score-bar-fill" style="width:{bar_pct}%"></div></div>
                <div style="font-size:0.8rem;color:#555;margin-top:0.5rem;line-height:1.5;">{overview_short}</div>
            </div>
            """, unsafe_allow_html=True)

# 푸터
st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown(
    '<div style="text-align:center;font-size:0.75rem;color:#333;font-family:\'DM Sans\',sans-serif;">'
    'TMDb 5000 Dataset · Content-Based + Weighted Rating Hybrid</div>',
    unsafe_allow_html=True
)
