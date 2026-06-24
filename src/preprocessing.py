"""
preprocessing.py
----------------
Carrega e limpa o dataset da Steam.

Suporta duas origens de dados:
  1. Arquivo local  → usado no desenvolvimento (Spyder, VS Code, etc.)
  2. URL remota     → usado no Streamlit Cloud (lê direto do GitHub)

O CSV slim (games_slim.csv) contém apenas as 13 colunas necessárias,
reduzindo o tamanho de 383MB → 45MB e eliminando a necessidade de Git LFS.

Estrutura do CSV slim (sem deslocamento de colunas — já foi corrigido):
    steam_id, game_name, Tags, Genres, Categories, Price, Positive,
    Negative, Metacritic score, Average playtime forever,
    Release date, Developers, Header image
"""

import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# URL do dataset no GitHub (raw) — atualizar com seu usuário após o push
# ---------------------------------------------------------------------------
GITHUB_USER = "Nickolas3211"
GITHUB_REPO = "steam-recommender"
GITHUB_BRANCH = "main"
DATA_FILENAME = "data/games_slim.csv"

GITHUB_RAW_URL = (
    f"https://raw.githubusercontent.com/"
    f"{GITHUB_USER}/{GITHUB_REPO}/{GITHUB_BRANCH}/{DATA_FILENAME}"
)

# ---------------------------------------------------------------------------
# Colunas do CSV slim (já corretas, sem shift)
# ---------------------------------------------------------------------------
SLIM_COLS = [
    "steam_id", "game_name", "Tags", "Genres", "Categories",
    "Price", "Positive", "Negative", "Metacritic score",
    "Average playtime forever", "Release date", "Developers", "Header image",
]

MIN_REVIEWS = 10


def load_raw(source: str | None = None) -> pd.DataFrame:
    """
    Carrega o CSV slim.

    Parameters
    ----------
    source : str | None
        Caminho local ou URL. Se None, tenta local primeiro, depois GitHub.

    Returns
    -------
    pd.DataFrame
    """
    import os

    # Determinar origem
    if source is None:
        # Tentar local primeiro (desenvolvimento)
        local_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "data", "games_slim.csv"
        )
        if os.path.exists(local_path):
            source = local_path
            logger.info(f"Carregando arquivo local: {local_path}")
        else:
            source = GITHUB_RAW_URL
            logger.info(f"Arquivo local não encontrado. Carregando do GitHub...")

    df = pd.read_csv(source, low_memory=False)

    # Garantir que as colunas esperadas existem
    missing = [c for c in SLIM_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Colunas ausentes no CSV: {missing}")

    logger.info(f"Dataset carregado: {len(df):,} jogos.")
    return df


def clean(df: pd.DataFrame, min_reviews: int = MIN_REVIEWS) -> pd.DataFrame:
    """
    Aplica filtros de qualidade e engenharia de colunas derivadas.

    Passos:
    1. Remove jogos sem Tags.
    2. Normaliza colunas numéricas.
    3. Cria 'total_reviews' e 'review_score'.
    4. Filtra por mínimo de reviews.
    5. Reseta o índice (necessário para indexar a feature matrix).
    """
    # --- Tipos numéricos ---
    numeric_cols = ["Price", "Positive", "Negative",
                    "Metacritic score", "Average playtime forever"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # --- Features derivadas ---
    df["total_reviews"] = df["Positive"] + df["Negative"]
    df["review_score"] = np.where(
        df["total_reviews"] > 0,
        df["Positive"] / df["total_reviews"],
        0.0,
    )

    # --- Filtros de qualidade ---
    before = len(df)
    df = df[df["Tags"].notna()].copy()
    logger.info(f"Removidos {before - len(df):,} jogos sem Tags.")

    before = len(df)
    df = df[df["total_reviews"] >= min_reviews].copy()
    logger.info(f"Removidos {before - len(df):,} jogos com < {min_reviews} reviews.")

    df = df.reset_index(drop=True)
    logger.info(f"Dataset limpo: {len(df):,} jogos.")
    return df
