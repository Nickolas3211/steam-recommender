"""
app.py — Steam Game Recommender com dados ao vivo da Steam API
Roda localmente:  streamlit run app.py
Deploy na nuvem:  https://share.streamlit.io
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
import pandas as pd
import numpy as np

from preprocessing import load_raw, clean
from features import build_feature_matrix
from recommender import ContentRecommender
from steam_api import fetch_live_data

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Steam Game Recommender",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# ESTILOS
# ─────────────────────────────────────────────
st.markdown("""
<style>
.stApp { background: #0f1117; }
[data-testid="stSidebar"] { background: #16181f; }

.game-card {
    background: linear-gradient(145deg, #1a1d27, #12141c);
    border: 1px solid #2a2d3a; border-radius: 12px;
    overflow: hidden; transition: transform 0.2s, border-color 0.2s; height: 100%;
}
.game-card:hover { transform: translateY(-4px); border-color: #4a90d9; }
.game-card img { width: 100%; height: 140px; object-fit: cover; display: block; }
.card-body { padding: 12px 14px 14px; }
.card-title {
    font-size: 0.88rem; font-weight: 700; color: #e8eaf0;
    margin: 0 0 6px; line-height: 1.3; height: 2.6em; overflow: hidden;
    display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical;
}
.card-genres { font-size: 0.72rem; color: #8b8fa8; margin-bottom: 8px;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.sim-bar-wrap { background: #252836; border-radius: 4px; height: 6px;
    overflow: hidden; margin-bottom: 6px; }
.sim-bar { height: 100%; border-radius: 4px;
    background: linear-gradient(90deg, #1565c0, #42a5f5); }
.sim-label { display: flex; justify-content: space-between;
    font-size: 0.68rem; color: #6b7194; }
.badge { display: inline-block; background: #1e2233; color: #8ecfff;
    border-radius: 4px; padding: 2px 7px; font-size: 0.65rem; margin: 2px 2px 0 0; }
.live-badge {
    display: inline-block; background: #1b3a1b; color: #66bb6a;
    border: 1px solid #2e7d32; border-radius: 6px;
    padding: 2px 8px; font-size: 0.68rem; margin-left: 8px; vertical-align: middle;
}
.offline-badge {
    display: inline-block; background: #2a1f1f; color: #ef9a9a;
    border: 1px solid #c62828; border-radius: 6px;
    padding: 2px 8px; font-size: 0.68rem; margin-left: 8px; vertical-align: middle;
}
.query-card {
    background: linear-gradient(135deg, #0d47a1, #1565c0, #1976d2);
    border-radius: 16px; overflow: hidden; display: flex;
    margin-bottom: 20px; border: 1px solid #1e88e5;
    box-shadow: 0 8px 32px rgba(21,101,192,0.3);
}
.query-card img { width: 300px; min-width: 300px; object-fit: cover; }
.query-info { padding: 24px 28px; flex: 1; }
.query-title { font-size: 1.6rem; font-weight: 800; color: #fff; margin-bottom: 8px; }
.query-meta { color: #90caf9; font-size: 0.85rem; margin-bottom: 12px; }
.query-tags { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }
.qtag { background: rgba(255,255,255,0.12); color: #e3f2fd;
    border-radius: 6px; padding: 3px 10px; font-size: 0.72rem; }

/* Live data panel */
.live-panel {
    background: #0d1f0d;
    border: 1px solid #2e7d32;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 20px;
}
.live-panel-header {
    font-size: 0.78rem; color: #66bb6a; font-weight: 700;
    margin-bottom: 12px; display: flex; align-items: center; gap: 8px;
}
.live-row { display: flex; flex-wrap: wrap; gap: 20px; }
.live-item { flex: 1; min-width: 120px; }
.live-item-label { font-size: 0.65rem; color: #4caf50; text-transform: uppercase;
    letter-spacing: 0.06em; margin-bottom: 3px; }
.live-item-val { font-size: 1.0rem; font-weight: 700; color: #e8f5e9; }
.live-item-sub { font-size: 0.7rem; color: #81c784; margin-top: 1px; }
.discount-tag {
    display: inline-block; background: #4caf50; color: #000;
    border-radius: 4px; padding: 1px 6px; font-size: 0.7rem;
    font-weight: 800; margin-left: 6px;
}
.steam-link { display: inline-block; margin-top: 10px; background: #1b2838;
    color: #c6d4df !important; padding: 6px 14px; border-radius: 6px;
    text-decoration: none !important; font-size: 0.78rem; border: 1px solid #2a475e; }
.section-title { font-size: 1.1rem; font-weight: 700; color: #c5cae9;
    margin: 24px 0 14px; padding-bottom: 8px; border-bottom: 2px solid #1e2233; }
.score-positive { color: #66bb6a; font-weight: 700; }
.score-mixed    { color: #ffa726; font-weight: 700; }
.score-negative { color: #ef5350; font-weight: 700; }
.metric-box { background: #1a1d27; border-radius: 10px; padding: 14px 18px;
    border: 1px solid #252836; text-align: center; }
.metric-val { font-size: 1.5rem; font-weight: 800; color: #e8eaf0; }
.metric-lbl { font-size: 0.72rem; color: #6b7194; margin-top: 2px; }
[data-testid="stToolbar"] { display: none; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def cover_url(steam_id) -> str:
    return f"https://cdn.cloudflare.steamstatic.com/steam/apps/{int(steam_id)}/header.jpg"

def store_url(steam_id) -> str:
    return f"https://store.steampowered.com/app/{int(steam_id)}"

def score_color(score: float) -> str:
    if score >= 0.75: return "score-positive"
    if score >= 0.50: return "score-mixed"
    return "score-negative"

def fmt_reviews(n: int) -> str:
    if n is None: return "N/A"
    if n >= 1_000_000: return f"{n/1_000_000:.1f}M"
    if n >= 1_000:     return f"{n/1_000:.1f}k"
    return str(n)

def render_game_card(row: pd.Series, rank: int) -> str:
    sim      = float(row.get("similarity_score", 0))
    score    = float(row.get("review_score", 0))
    price    = float(row.get("Price", 0))
    name     = str(row.get("game_name", ""))
    genres   = ", ".join(g.strip() for g in str(row.get("Genres","")).split(",")[:3])
    sid      = int(row.get("steam_id", 0))
    top_tags = [t.strip() for t in str(row.get("Tags","")).split(",")[:4]]
    badges   = "".join(f'<span class="badge">{t}</span>' for t in top_tags if t)
    price_str = "Grátis" if price == 0 else f"${price:.2f}"
    bar_w    = int(sim * 100)
    return f"""
<div class="game-card">
  <a href="{store_url(sid)}" target="_blank">
    <img src="{cover_url(sid)}" alt="{name}"
         onerror="this.src='https://placehold.co/460x215/1a1d27/4a90d9?text=No+Image'"/>
  </a>
  <div class="card-body">
    <div class="card-title">#{rank} {name}</div>
    <div class="card-genres">{genres}</div>
    <div class="sim-bar-wrap"><div class="sim-bar" style="width:{bar_w}%"></div></div>
    <div class="sim-label"><span>Similaridade</span><span><b>{sim:.3f}</b></span></div>
    <div style="margin-top:6px;font-size:0.72rem;">
      <span class="{score_color(score)}">{score:.0%}</span>
      <span style="color:#555;margin:0 4px">|</span>
      <span style="color:#8b8fa8">{price_str}</span>
    </div>
    <div style="margin-top:8px">{badges}</div>
  </div>
</div>"""

def render_query_card(info: pd.Series) -> str:
    name  = str(info.get("game_name", ""))
    sid   = int(info.get("steam_id", 0))
    score = float(info.get("review_score", 0))
    total = int(info.get("total_reviews", 0))
    price = float(info.get("Price", 0))
    dev   = str(info.get("Developers", ""))
    rel   = str(info.get("Release date", ""))
    genres = str(info.get("Genres", ""))
    top_tags = [t.strip() for t in str(info.get("Tags","")).split(",")[:12]]
    tags_html = "".join(f'<span class="qtag">{t}</span>' for t in top_tags if t)
    price_str = "Grátis" if price == 0 else f"${price:.2f}"
    return f"""
<div class="query-card">
  <a href="{store_url(sid)}" target="_blank">
    <img src="{cover_url(sid)}" alt="{name}"
         onerror="this.src='https://placehold.co/460x215/0d47a1/ffffff?text=No+Image'"/>
  </a>
  <div class="query-info">
    <div class="query-title">{name}</div>
    <div class="query-meta">
      {dev} &nbsp;·&nbsp; {rel} &nbsp;·&nbsp;
      <span class="{score_color(score)}">{score:.0%} positivo</span>
      &nbsp;({total:,} avaliações) &nbsp;·&nbsp; {price_str}
    </div>
    <div style="font-size:0.8rem;color:#90caf9;margin-bottom:8px">{genres}</div>
    <div class="query-tags">{tags_html}</div>
    <a class="steam-link" href="{store_url(sid)}" target="_blank">🔗 Ver na Steam</a>
  </div>
</div>"""

def render_live_panel(live: dict) -> str:
    """Renderiza o painel verde com dados ao vivo da Steam API."""
    # Preço
    price_str = live.get("price_formatted") or "N/A"
    discount  = live.get("discount_pct", 0)
    discount_tag = f'<span class="discount-tag">-{discount}%</span>' if discount else ""

    # Reviews
    pos_pct   = live.get("positive_pct")
    score_lbl = live.get("score_label_pt") or "—"
    total_rev = live.get("total_reviews")
    total_pos = live.get("total_positive")
    pct_str   = f"{pos_pct:.1%}" if pos_pct is not None else "N/A"
    rev_str   = fmt_reviews(total_rev)
    pos_str   = fmt_reviews(total_pos)

    # Metacritic
    mc = live.get("metacritic")
    mc_str = str(int(mc)) if mc else "N/A"
    mc_color = (
        "#66bb6a" if mc and mc >= 75 else
        "#ffa726" if mc and mc >= 50 else
        "#ef5350" if mc else "#8b8fa8"
    )

    # Gêneros ao vivo
    genres_live = ", ".join(live.get("genres_live", [])) or "—"

    return f"""
<div class="live-panel">
  <div class="live-panel-header">
    <span>🟢</span>
    <span>DADOS AO VIVO — Steam Store</span>
    <span style="color:#2e7d32;font-weight:400;margin-left:4px">
      (atualizado agora via Steam API)
    </span>
  </div>
  <div class="live-row">
    <div class="live-item">
      <div class="live-item-label">Preço Atual</div>
      <div class="live-item-val">{price_str}{discount_tag}</div>
      <div class="live-item-sub">USD · Steam Store</div>
    </div>
    <div class="live-item">
      <div class="live-item-label">Avaliação Geral</div>
      <div class="live-item-val">{pct_str} positivas</div>
      <div class="live-item-sub">{score_lbl}</div>
    </div>
    <div class="live-item">
      <div class="live-item-label">Total de Reviews</div>
      <div class="live-item-val">{rev_str}</div>
      <div class="live-item-sub">{pos_str} positivas</div>
    </div>
    <div class="live-item">
      <div class="live-item-label">Metacritic</div>
      <div class="live-item-val" style="color:{mc_color}">{mc_str}</div>
      <div class="live-item-sub">Score da crítica</div>
    </div>
    <div class="live-item">
      <div class="live-item-label">Gêneros (Steam)</div>
      <div class="live-item-val" style="font-size:0.8rem">{genres_live}</div>
      <div class="live-item-sub">Classificação oficial</div>
    </div>
  </div>
</div>"""


# ─────────────────────────────────────────────
# CACHE
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="⚙️  Carregando modelo... (~30s na primeira vez)")
def load_system():
    df_raw = load_raw()
    df = clean(df_raw)
    feature_matrix, encoders, _ = build_feature_matrix(df)
    recommender = ContentRecommender(df, feature_matrix)
    return df, feature_matrix, recommender


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎮 Steam Recommender")
    st.markdown(
        "Sistema **Content-Based** usando Similaridade do Cosseno "
        "sobre Tags, Gêneros e Categorias da Steam."
    )
    st.divider()

    top_n = st.slider("Número de recomendações", 5, 20, 10)
    st.markdown("**Filtros**")
    min_score = st.slider("Score mínimo", 0.0, 1.0, 0.0, 0.05, format="%.0f%%")
    max_price = st.slider("Preço máximo (USD)", 0, 70, 70)
    exclude_dev = st.checkbox("Excluir mesmo desenvolvedor", value=False)

    st.divider()
    st.markdown("**Exemplos rápidos**")
    examples = [
        "Hollow Knight", "Stardew Valley", "Hades", "Terraria",
        "Celeste", "Dead Cells", "Cuphead", "Into the Breach",
        "Slay the Spire", "Vampire Survivors",
    ]
    selected_example = st.selectbox("Escolher →", [""] + examples)

    st.divider()
    st.caption(
        "📦 Dataset: Steam Games (Kaggle)  \n"
        "🔢 56k+ jogos · 541 features  \n"
        "🌐 Preço e reviews: Steam API (ao vivo)"
    )


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
st.markdown("# 🎮 Steam Game Recommender")
st.markdown(
    "Digite o nome de um jogo para encontrar títulos similares. "
    "Dados ao vivo de preço e avaliações via Steam API."
)
st.divider()

df, feature_matrix, recommender = load_system()

# Campo de busca
query = st.text_input(
    "🔍 Buscar jogo",
    value=selected_example if selected_example else "",
    placeholder="Ex: Hollow Knight, Stardew Valley, Hades...",
)

# Autocomplete
if query and len(query) >= 2:
    suggestions = recommender.search(query, top_k=6)
    exact = any(s.lower() == query.lower() for s in suggestions)
    if suggestions and not exact:
        with st.expander(f"💡 {len(suggestions)} sugestão(ões)"):
            for s in suggestions:
                st.markdown(f"- **{s}**")

# ─── Resultado ───
if query:
    try:
        game_info = recommender.get_game_info(query)
        sid = int(game_info["steam_id"])

        # Card estático do jogo
        st.markdown('<div class="section-title">📌 Jogo Consultado</div>',
                    unsafe_allow_html=True)
        st.markdown(render_query_card(game_info), unsafe_allow_html=True)

        # ── Botão de dados ao vivo ──
        col_btn, col_status = st.columns([2, 8])
        with col_btn:
            fetch_live = st.button(
                "🔄 Atualizar dados ao vivo",
                help="Busca preço e avaliações em tempo real direto da Steam API",
                type="primary",
            )

        # Estado da sessão: guardar último resultado ao vivo por jogo
        live_key = f"live_{sid}"
        if fetch_live:
            with st.spinner("Conectando à Steam API..."):
                live_data = fetch_live_data(sid)
            st.session_state[live_key] = live_data

        # Exibir painel ao vivo se disponível
        if live_key in st.session_state:
            live = st.session_state[live_key]
            if live.get("any_success"):
                st.markdown(render_live_panel(live), unsafe_allow_html=True)
            else:
                st.warning(
                    "⚠️ Não foi possível conectar à Steam API agora. "
                    "Verifique sua conexão ou tente novamente em instantes."
                )
        else:
            with col_status:
                st.caption("💡 Clique em **Atualizar dados ao vivo** para buscar preço e avaliações em tempo real.")

        # Métricas do dataset (históricas)
        st.markdown('<div class="section-title" style="margin-top:12px">📊 Dados do Dataset</div>',
                    unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        score_val  = float(game_info.get("review_score", 0))
        total_rev  = int(game_info.get("total_reviews", 0))
        metacritic = float(game_info.get("Metacritic score", 0))
        playtime   = float(game_info.get("Average playtime forever", 0))

        for col, val, lbl, sc in [
            (m1, f"{score_val:.0%}", "Score (dataset)", score_color(score_val)),
            (m2, f"{fmt_reviews(total_rev)}", "Total Reviews", ""),
            (m3, str(int(metacritic)) if metacritic > 0 else "N/A", "Metacritic (dataset)", ""),
            (m4, f"{playtime/60:.0f}h" if playtime >= 60 else f"{int(playtime)}min", "Tempo Médio de Jogo", ""),
        ]:
            with col:
                st.markdown(f"""<div class="metric-box">
                    <div class="metric-val {sc}">{val}</div>
                    <div class="metric-lbl">{lbl}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("")

        # Recomendações
        recs = recommender.recommend(query, top_n=top_n * 3,
                                     exclude_same_developer=exclude_dev)
        if min_score > 0:
            recs = recs[recs["review_score"] >= min_score]
        if max_price < 70:
            recs = recs[recs["Price"] <= max_price]
        recs = recs.head(top_n).reset_index(drop=True)

        st.markdown(f'<div class="section-title">🎯 {len(recs)} Jogos Similares</div>',
                    unsafe_allow_html=True)

        if recs.empty:
            st.warning("Nenhum resultado com os filtros aplicados. Tente relaxar os filtros.")
        else:
            COLS_PER_ROW = 5
            for row_start in range(0, len(recs), COLS_PER_ROW):
                row_slice = recs.iloc[row_start:row_start + COLS_PER_ROW]
                cols = st.columns(COLS_PER_ROW)
                for col_i, (_, game_row) in enumerate(row_slice.iterrows()):
                    with cols[col_i]:
                        st.markdown(
                            render_game_card(game_row, row_start + col_i + 1),
                            unsafe_allow_html=True,
                        )

            with st.expander("📊 Ver tabela detalhada"):
                show = recs[["game_name", "Genres", "review_score",
                              "Price", "total_reviews", "similarity_score"]].copy()
                show.columns = ["Jogo", "Gêneros", "Score", "Preço (USD)",
                                 "Reviews", "Similaridade"]
                show["Score"] = show["Score"].map("{:.0%}".format)
                show["Similaridade"] = show["Similaridade"].map("{:.4f}".format)
                show["Preço (USD)"] = show["Preço (USD)"].apply(
                    lambda x: "Grátis" if x == 0 else f"${x:.2f}")
                st.dataframe(show, use_container_width=True, hide_index=True)

    except ValueError as e:
        st.error(f"❌ {e}")
        st.info("Dica: verifique o nome ou use a lista de exemplos na barra lateral.")

else:
    st.markdown('<div class="section-title">🔥 Jogos em destaque</div>',
                unsafe_allow_html=True)
    preview = ["Hollow Knight", "Stardew Valley", "Hades", "Terraria", "Celeste"]
    cols = st.columns(5)
    for i, game in enumerate(preview):
        try:
            info = recommender.get_game_info(game)
            sid = int(info["steam_id"])
            score = float(info.get("review_score", 0))
            with cols[i]:
                st.markdown(f"""
<div class="game-card">
  <a href="{store_url(sid)}" target="_blank">
    <img src="{cover_url(sid)}" alt="{game}"
         onerror="this.src='https://placehold.co/460x215/1a1d27/4a90d9?text=No+Image'"/>
  </a>
  <div class="card-body">
    <div class="card-title">{game}</div>
    <div class="sim-label">
      <span style="color:#8b8fa8">Avaliação</span>
      <span class="{score_color(score)}">{score:.0%}</span>
    </div>
  </div>
</div>""", unsafe_allow_html=True)
        except Exception:
            pass
    st.markdown("")
    st.info("💡 Use o campo de busca acima ou selecione um exemplo na barra lateral.")
