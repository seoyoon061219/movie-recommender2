"""
🎬 TMDb 영화 추천 시스템
콘텐츠 기반 + 인기도/평점 가중치 하이브리드
데이터: tmdb_5000_movies.csv + tmdb_5000_credits.csv
"""

import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ─────────────────────────────────────────
# 1. 데이터 로드 & 병합
# ─────────────────────────────────────────

def load_data(movies_path: str, credits_path: str) -> pd.DataFrame:
    movies  = pd.read_csv(movies_path)
    credits = pd.read_csv(credits_path)
    credits.rename(columns={'movie_id': 'id'}, inplace=True)
    df = movies.merge(credits, on='id')
    return df


# ─────────────────────────────────────────
# 2. 전처리 함수
# ─────────────────────────────────────────

def parse_names(obj, top: int = 5) -> list:
    """JSON 문자열 → 이름 리스트 (공백 제거해 하나의 토큰으로 처리)"""
    try:
        return [d['name'].replace(' ', '') for d in ast.literal_eval(obj)[:top]]
    except Exception:
        return []


def get_director(crew_str: str) -> str:
    """crew JSON에서 감독 이름 추출"""
    try:
        for person in ast.literal_eval(crew_str):
            if person.get('job') == 'Director':
                return person['name'].replace(' ', '')
    except Exception:
        pass
    return ''


def make_soup(row) -> str:
    """
    모든 콘텐츠 특성을 하나의 문자열(soup)로 합침
    - 감독은 x2 가중치 (중요도 반영)
    - overview는 단어 단위로 분리
    """
    overview_words = row['overview'].split() if isinstance(row['overview'], str) else []
    parts = (
        row['genres_list']
        + row['keywords_list']
        + row['cast_list']
        + [row['director']] * 2
        + overview_words
    )
    return ' '.join(parts)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """결측치 처리 + 특성 컬럼 생성"""
    df = df.dropna(subset=['overview']).reset_index(drop=True)

    df['genres_list']   = df['genres'].apply(parse_names)
    df['keywords_list'] = df['keywords'].apply(lambda x: parse_names(x, top=10))
    df['cast_list']     = df['cast'].apply(lambda x: parse_names(x, top=3))
    df['director']      = df['crew'].apply(get_director)
    df['soup']          = df.apply(make_soup, axis=1)

    return df


# ─────────────────────────────────────────
# 3. 모델 구축
# ─────────────────────────────────────────

def build_model(df: pd.DataFrame):
    """CountVectorizer + 코사인 유사도 행렬 생성"""
    cv = CountVectorizer(stop_words='english')
    matrix = cv.fit_transform(df['soup'])
    cosine_sim = cosine_similarity(matrix, matrix)
    # 제목 → 인덱스 매핑
    indices = pd.Series(df.index, index=df['title']).drop_duplicates()
    return cosine_sim, indices


# ─────────────────────────────────────────
# 4. 가중 평점 계산 (인기도 보정)
# ─────────────────────────────────────────

def compute_weighted_score(df: pd.DataFrame, percentile: float = 0.70) -> pd.Series:
    """
    IMDB 방식 가중 평점 (Bayesian Average)
    score = (v/(v+m)) * R + (m/(v+m)) * C
      v = 해당 영화 vote_count
      m = 최소 기준 vote_count (하위 70% 컷)
      R = 해당 영화 vote_average
      C = 전체 평균 vote_average
    """
    m = df['vote_count'].quantile(percentile)
    C = df['vote_average'].mean()

    def score(row):
        v = row['vote_count']
        R = row['vote_average']
        return (v / (v + m)) * R + (m / (v + m)) * C

    return df.apply(score, axis=1)


# ─────────────────────────────────────────
# 5. 추천 함수 (하이브리드)
# ─────────────────────────────────────────

def recommend(
    title: str,
    df: pd.DataFrame,
    cosine_sim: np.ndarray,
    indices: pd.Series,
    top_n: int = 10,
    content_weight: float = 0.7,
    score_weight: float = 0.3,
) -> pd.DataFrame:
    """
    하이브리드 추천:
      최종 점수 = content_weight × 콘텐츠 유사도
               + score_weight  × 정규화된 가중 평점

    Args:
        title          : 기준 영화 제목 (영어)
        df             : 전처리된 데이터프레임
        cosine_sim     : 코사인 유사도 행렬
        indices        : 제목 → 인덱스 매핑
        top_n          : 추천 영화 수
        content_weight : 콘텐츠 유사도 가중치 (기본 0.7)
        score_weight   : 인기도/평점 가중치 (기본 0.3)

    Returns:
        추천 영화 DataFrame (title, genres, director, vote_average, hybrid_score)
    """
    # 제목 검색
    if title not in indices:
        close = [t for t in indices.index if title.lower() in t.lower()]
        if close:
            print(f"  ⚠️  '{title}' 없음 → 가장 유사한 제목: {close[:3]}")
            title = close[0]
        else:
            print(f"  ❌ '{title}'을(를) 데이터셋에서 찾을 수 없습니다.")
            return pd.DataFrame()

    idx = indices[title]

    # 콘텐츠 유사도 점수
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n * 5]  # 후보군 확보

    movie_indices = [i[0] for i in sim_scores]
    sim_values    = [i[1] for i in sim_scores]

    # 후보 DataFrame
    candidates = df.iloc[movie_indices].copy()
    candidates['content_score'] = sim_values

    # 가중 평점 정규화 (0~1)
    ws = compute_weighted_score(candidates)
    candidates['weighted_score'] = ws
    ws_min, ws_max = ws.min(), ws.max()
    if ws_max > ws_min:
        candidates['weighted_score_norm'] = (ws - ws_min) / (ws_max - ws_min)
    else:
        candidates['weighted_score_norm'] = 0.5

    # 하이브리드 점수
    candidates['hybrid_score'] = (
        content_weight * candidates['content_score']
        + score_weight  * candidates['weighted_score_norm']
    )

    # 최종 정렬 & 컬럼 정리
    result = (
        candidates
        .sort_values('hybrid_score', ascending=False)
        .head(top_n)
        [['title', 'genres_list', 'director', 'vote_average', 'vote_count', 'hybrid_score']]
        .rename(columns={
            'genres_list': 'genres',
            'vote_average':'rating',
            'vote_count':  'votes',
        })
    )
    result['genres'] = result['genres'].apply(lambda g: ', '.join(g))
    result['hybrid_score'] = result['hybrid_score'].round(4)
    result = result.reset_index(drop=True)
    result.index += 1  # 1부터 시작
    return result


# ─────────────────────────────────────────
# 6. 실행 예시
# ─────────────────────────────────────────

if __name__ == '__main__':
    # ── 경로 설정 ──────────────────────────
    MOVIES_PATH  = 'tmdb_5000_movies.csv'
    CREDITS_PATH = 'tmdb_5000_credits.csv'

    # ── 로드 & 전처리 ──────────────────────
    print("📂 데이터 로드 중...")
    df_raw = load_data(MOVIES_PATH, CREDITS_PATH)

    print("🔧 전처리 중...")
    df = preprocess(df_raw)

    print("🧮 유사도 행렬 계산 중...")
    cosine_sim, indices = build_model(df)

    print(f"✅ 준비 완료! 총 {len(df)}편의 영화\n")

    # ── 추천 테스트 ────────────────────────
    test_movies = ['Avatar', 'The Dark Knight', 'Inception', 'The Godfather']

    for movie in test_movies:
        print(f"{'─'*55}")
        print(f"🎬 '{movie}' 기반 추천 TOP 10")
        print(f"{'─'*55}")
        result = recommend(movie, df, cosine_sim, indices, top_n=10)
        if not result.empty:
            print(result.to_string())
        print()
