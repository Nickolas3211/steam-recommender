# 🎮 Steam Game Recommender — Content-Based Filtering

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://nickolas3211-steam-recommender.streamlit.app)

Sistema de recomendação de jogos da Steam baseado em conteúdo (**Content-Based Filtering**), usando Similaridade do Cosseno sobre Tags, Gêneros e Categorias — com dados ao vivo de preço e avaliações via Steam API.

---

## 🧠 Problema de Negócio

A Steam possui mais de 50.000 jogos ativos. Usuários ficam paralisados pela abundância de escolhas — o chamado *paradoxo da escolha*. Um sistema de recomendação baseado em conteúdo resolve isso: dado um jogo que você gostou, encontra os mais similares em gênero, mecânicas e perfil de avaliação, sem precisar do histórico de outros usuários.

---

## 🏗️ Arquitetura

```
steam-recommender/
├── app.py                      # Interface Streamlit (local + Streamlit Cloud)
├── src/
│   ├── preprocessing.py        # Limpeza e filtragem de qualidade
│   ├── features.py             # MultiLabelBinarizer + matriz esparsa
│   ├── recommender.py          # Motor de recomendação (Cosine Similarity)
│   ├── steam_api.py            # Integração com a Steam Web API (tempo real)
│   └── evaluation.py           # ILD, Genre Coherence, Catalog Coverage
├── data/
│   └── games_slim.csv          # Dataset reduzido (13 colunas, 45 MB)
├── .streamlit/
│   └── config.toml             # Tema dark + configurações do servidor
├── requirements.txt
└── README.md
```

---

## ⚙️ Decisões Técnicas

| Decisão | Justificativa |
|---|---|
| `MultiLabelBinarizer` | Tags são multi-label — `get_dummies` trataria `"Action,RPG"` como valor único em vez de duas colunas separadas |
| Similaridade do Cosseno | Ignora a magnitude do vetor, evitando viés entre jogos indie (poucas tags) e AAA (muitas tags) |
| Matriz esparsa `scipy.sparse` | ~97% dos valores são zero — reduz o uso de memória em ~98% |
| Pesos por feature | Tags (×2.0) > Gêneros (×1.5) > Categorias (×1.0) > Numéricas (×0.5) |
| Steam API on-demand | Busca preço e avaliações em tempo real apenas para o jogo consultado, respeitando o rate limit |
| CSV slim (45 MB) | CSV original tem 383 MB — extraídas apenas as 13 colunas necessárias para caber no GitHub |
| `@st.cache_resource` | Modelo carregado uma única vez por sessão, sem reprocessar a cada interação |

---

## 📐 Pipeline

```
games_slim.csv (45 MB)
       ↓  preprocessing.py — filtragem de qualidade (tags + mín. 10 reviews)
DataFrame limpo (~56.661 jogos)
       ↓  features.py — MultiLabelBinarizer + MinMaxScaler + ponderação
Matriz esparsa (56.661 × 541 features)
       ↓  recommender.py — Cosine Similarity on-demand
Top-N jogos mais similares
       ↓  steam_api.py — preço e reviews ao vivo (apenas para o jogo consultado)
       ↓  app.py — interface Streamlit com capas, scores e links para a Steam
```

---

## 🚀 Rodar Localmente

```bash
# 1. Clonar o repositório
git clone https://github.com/Nickolas3211/steam-recommender.git
cd steam-recommender

# 2. Instalar dependências
pip install -r requirements.txt

# 3. Rodar
streamlit run app.py
```

Acesse em `http://localhost:8501`

---

## 📊 Dataset

[Steam Games Dataset — Kaggle](https://www.kaggle.com/datasets/fronkongames/steam-games-dataset)

| | Original | Slim (neste repo) | Após filtragem |
|---|---|---|---|
| Jogos | 125.855 | 125.855 | 56.661 |
| Colunas | 39 | 13 | 13 + 2 derivadas |
| Tamanho | 383 MB | 45 MB | — |

---

## 🛠️ Stack

`Python` · `Pandas` · `Scikit-learn` · `Scipy` · `Streamlit` · `Requests`

---

---

# 🎮 Steam Game Recommender — Content-Based Filtering *(English)*

A Steam game recommendation system using **Content-Based Filtering** with Cosine Similarity over Tags, Genres and Categories — plus live price and review data via the Steam Web API.

**Key design decisions:**
- `MultiLabelBinarizer` for multi-label tag encoding (not `get_dummies`)
- Sparse matrix (~97% sparsity, ~98% memory savings vs dense)
- On-demand cosine similarity — no pre-computed N×N matrix stored in memory
- Feature weighting: Tags (2×) > Genres (1.5×) > Categories (1×) > Numeric (0.5×)
- Steam API called on-demand for the queried game only (respects rate limits)
- Slim CSV (45 MB) extracted from 383 MB original to fit GitHub without Git LFS

**Run locally:** `pip install -r requirements.txt && streamlit run app.py`

*Portfolio project — Data Science · Recommendation Systems*
