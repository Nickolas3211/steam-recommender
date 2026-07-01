# 🎮 Steam Game Recommender — Content-Based Filtering

**[🇧🇷 Português](#-português) | [🇺🇸 English](#-english)**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://nickolas3211-steam-recommender.streamlit.app)

---

## 🇧🇷 Português

Sistema de recomendação de jogos da Steam baseado em conteúdo (**Content-Based Filtering**), usando Similaridade do Cosseno sobre Tags, Gêneros e Categorias — com dados ao vivo de preço e avaliações via Steam API.

### Índice

- [Problema de Negócio](#-problema-de-negócio)
- [Arquitetura](#️-arquitetura)
- [Decisões Técnicas](#️-decisões-técnicas)
- [Pipeline](#-pipeline)
- [Rodar Localmente](#-rodar-localmente)
- [Dataset](#-dataset)
- [Stack](#️-stack)

### 🧠 Problema de Negócio

A Steam possui mais de 50.000 jogos ativos. Usuários ficam paralisados pela abundância de escolhas — o chamado *paradoxo da escolha*. Um sistema de recomendação baseado em conteúdo resolve isso: dado um jogo que você gostou, encontra os mais similares em gênero, mecânicas e perfil de avaliação, sem precisar do histórico de outros usuários.

### 🏗️ Arquitetura

```
steam-recommender/
├── app.py
│   └── Interface web e experiência do usuário (Streamlit)
│
├── src/
│   ├── preprocessing.py
│   │   └── Limpeza, validação e preparação dos dados
│   │
│   ├── features.py
│   │   └── Engenharia de features e construção da matriz esparsa
│   │
│   ├── recommender.py
│   │   └── Sistema de recomendação baseado em Similaridade do Cosseno
│   │
│   ├── steam_api.py
│   │   └── Integração com a Steam API para dados em tempo real
│   │
│   └── evaluation.py
│       └── Métricas de avaliação das recomendações
│
├── data/
│   └── games_slim.csv
│       └── Dataset otimizado (~56 mil jogos)
│
├── .streamlit/
│   └── config.toml
│       └── Configurações da aplicação
│
├── requirements.txt
│   └── Dependências do projeto
│
└── README.md
    └── Documentação técnica
```

### ⚙️ Decisões Técnicas

| Decisão | Justificativa |
|---|---|
| `MultiLabelBinarizer` | Tags são multi-label — `get_dummies` trataria `"Action,RPG"` como valor único em vez de duas colunas separadas |
| Similaridade do Cosseno | Ignora a magnitude do vetor, evitando viés entre jogos indie (poucas tags) e AAA (muitas tags) |
| Matriz esparsa `scipy.sparse` | ~97% dos valores são zero — reduz o uso de memória em ~98% |
| Pesos por feature | Tags (×2.0) > Gêneros (×1.5) > Categorias (×1.0) > Numéricas (×0.5) |
| Steam API on-demand | Busca preço e avaliações em tempo real apenas para o jogo consultado, respeitando o rate limit |
| CSV slim (45 MB) | CSV original tem 383 MB — extraídas apenas as 13 colunas necessárias para caber no GitHub |
| `@st.cache_resource` | Modelo carregado uma única vez por sessão, sem reprocessar a cada interação |

### 📐 Pipeline

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

### 🚀 Rodar Localmente

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

### 📊 Dataset

[Steam Games Dataset — Kaggle](https://www.kaggle.com/datasets/fronkongames/steam-games-dataset)

| | Original | Slim (neste repo) | Após filtragem |
|---|---|---|---|
| Jogos | 125.855 | 125.855 | 56.661 |
| Colunas | 39 | 13 | 13 + 2 derivadas |
| Tamanho | 383 MB | 45 MB | — |

### 🛠️ Stack

`Python` · `Pandas` · `Scikit-learn` · `Scipy` · `Streamlit` · `Requests`

---

## 🇺🇸 English

A Steam game recommendation system based on **Content-Based Filtering**, using Cosine Similarity over Tags, Genres and Categories — with live price and review data via the Steam Web API.

### Table of Contents

- [Business Problem](#-business-problem)
- [Architecture](#️-architecture)
- [Technical Decisions](#️-technical-decisions)
- [Pipeline](#-pipeline-1)
- [Run Locally](#-run-locally)
- [Dataset](#-dataset-1)
- [Stack](#️-stack-1)

### 🧠 Business Problem

Steam has over 50,000 active games. Users get paralyzed by the sheer abundance of choice — the so-called *paradox of choice*. A content-based recommendation system solves this: given a game the user liked, it finds the most similar ones in genre, mechanics, and review profile, with no need for other users' interaction history.

### 🏗️ Architecture

```
steam-recommender/
├── app.py
│   └── Web interface and user experience (Streamlit)
│
├── src/
│   ├── preprocessing.py
│   │   └── Data cleaning, validation and preparation
│   │
│   ├── features.py
│   │   └── Feature engineering and sparse matrix construction
│   │
│   ├── recommender.py
│   │   └── Recommendation engine based on Cosine Similarity
│   │
│   ├── steam_api.py
│   │   └── Integration with the Steam API for real-time data
│   │
│   └── evaluation.py
│       └── Recommendation evaluation metrics
│
├── data/
│   └── games_slim.csv
│       └── Optimized dataset (~56k games)
│
├── .streamlit/
│   └── config.toml
│       └── App configuration
│
├── requirements.txt
│   └── Project dependencies
│
└── README.md
    └── Technical documentation
```

### ⚙️ Technical Decisions

| Decision | Rationale |
|---|---|
| `MultiLabelBinarizer` | Tags are multi-label — `get_dummies` would treat `"Action,RPG"` as a single value instead of two separate columns |
| Cosine Similarity | Ignores vector magnitude, avoiding bias between indie games (few tags) and AAA titles (many tags) |
| `scipy.sparse` matrix | ~97% of values are zero — cuts memory usage by ~98% |
| Feature weighting | Tags (×2.0) > Genres (×1.5) > Categories (×1.0) > Numerical (×0.5) |
| On-demand Steam API | Fetches price and reviews in real time only for the queried game, respecting the rate limit |
| Slim CSV (45 MB) | Original CSV is 383 MB — only the 13 required columns were extracted to fit within GitHub's limits |
| `@st.cache_resource` | Model is loaded once per session, avoiding reprocessing on every interaction |

### 📐 Pipeline

```
games_slim.csv (45 MB)
       ↓  preprocessing.py — quality filtering (tags + min. 10 reviews)
Clean DataFrame (~56,661 games)
       ↓  features.py — MultiLabelBinarizer + MinMaxScaler + weighting
Sparse matrix (56,661 × 541 features)
       ↓  recommender.py — on-demand Cosine Similarity
Top-N most similar games
       ↓  steam_api.py — live price and reviews (queried game only)
       ↓  app.py — Streamlit UI with covers, scores and Steam links
```

### 🚀 Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/Nickolas3211/steam-recommender.git
cd steam-recommender

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
streamlit run app.py
```

Access at `http://localhost:8501`

### 📊 Dataset

[Steam Games Dataset — Kaggle](https://www.kaggle.com/datasets/fronkongames/steam-games-dataset)

| | Original | Slim (this repo) | After filtering |
|---|---|---|---|
| Games | 125,855 | 125,855 | 56,661 |
| Columns | 39 | 13 | 13 + 2 derived |
| Size | 383 MB | 45 MB | — |

### 🛠️ Stack

`Python` · `Pandas` · `Scikit-learn` · `Scipy` · `Streamlit` · `Requests`

---

*Portfolio project — Data Science · Recommendation Systems*
