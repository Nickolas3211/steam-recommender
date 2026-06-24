"""
features.py
-----------
Transforma o dataset limpo numa matriz esparsa de features.

Pipeline:
    1. MultiLabelBinarizer  → Tags, Genres, Categories  (multi-label categórico)
    2. MinMaxScaler         → Price, review_score, Metacritic, Playtime (numérico)
    3. Ponderação por coluna → reflete importância semântica de cada feature
    4. scipy.sparse.hstack  → concatena tudo num único vetor por jogo

Por que esparso?
    452 tags × 56k jogos = 25 milhões de células. Cada jogo tem ~14 tags em média,
    logo ~97% das células seriam zero. A representação esparsa armazena apenas
    os valores não-nulos, reduzindo o uso de memória de ~200 MB para ~3 MB.
"""

import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pesos das features
# ---------------------------------------------------------------------------
# Tags: mais granulares e discriminativas (ex: "Roguelike", "Souls-like")
# Genres: mais amplos (ex: "Action", "RPG") — usados como confirmação de Tags
# Categories: mecânicas de jogo (ex: "Co-op", "VR") — contexto adicional
# Numéricas: sinal de qualidade/preço — peso menor para não dominar
WEIGHTS = {
    "tags":     2.0,
    "genres":   1.5,
    "categories": 1.0,
    "numeric":  0.5,
}

NUMERIC_COLS = [
    "Price",
    "review_score",
    "Metacritic score",
    "Average playtime forever",
]


def _parse_multilabel(series: pd.Series) -> pd.Series:
    """
    Converte coluna de string separada por vírgulas em lista de itens.

    Exemplo:
        "Action,RPG,Open World"  →  ["Action", "RPG", "Open World"]
        NaN                      →  []
    """
    return series.fillna("").apply(
        lambda x: [item.strip() for item in x.split(",") if item.strip()]
    )


def build_feature_matrix(df: pd.DataFrame):
    """
    Constrói a matriz esparsa de features e retorna os encoders para uso futuro.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset limpo (saída de preprocessing.clean).

    Returns
    -------
    feature_matrix : scipy.sparse.csr_matrix
        Matriz (n_jogos × n_features) com todas as features ponderadas.
    encoders : dict
        Dicionário com os objetos treinados {mlb_tags, mlb_genres, mlb_cats, scaler}.
    feature_names : list[str]
        Nomes de todas as features (útil para interpretabilidade).
    """
    logger.info("Iniciando construção da matriz de features...")

    # -----------------------------------------------------------------------
    # 1. Binarização Multi-label (MultiLabelBinarizer)
    # -----------------------------------------------------------------------
    # MultiLabelBinarizer é a ferramenta certa para listas de rótulos.
    # get_dummies() não funciona aqui: trataria "Action,RPG" como um único
    # rótulo string, gerando colunas inúteis como "Action,RPG" em vez de
    # colunas separadas "Action" e "RPG".
    #
    # sparse_output=True → a saída já é uma matriz esparsa scipy.
    # fit_transform():
    #   - fit: descobre o vocabulário completo de tags no corpus
    #   - transform: para cada jogo, gera vetor 0/1 indicando quais tags possui

    mlb_tags = MultiLabelBinarizer(sparse_output=True)
    mlb_genres = MultiLabelBinarizer(sparse_output=True)
    mlb_cats = MultiLabelBinarizer(sparse_output=True)

    tags_sparse = mlb_tags.fit_transform(_parse_multilabel(df["Tags"]))
    genres_sparse = mlb_genres.fit_transform(_parse_multilabel(df["Genres"]))
    cats_sparse = mlb_cats.fit_transform(_parse_multilabel(df.get("Categories", pd.Series())))

    logger.info(
        f"  Tags binarizadas: {tags_sparse.shape[1]} tags únicas\n"
        f"  Genres binarizados: {genres_sparse.shape[1]} géneros únicos\n"
        f"  Categories binarizadas: {cats_sparse.shape[1]} categorias únicas"
    )

    # -----------------------------------------------------------------------
    # 2. Normalização das features numéricas (MinMaxScaler)
    # -----------------------------------------------------------------------
    # MinMaxScaler mapeia cada coluna para [0, 1] dividindo pelo range.
    # Necessário porque Price (0–100 USD) e Playtime (0–300.000 min) estão
    # em escalas completamente diferentes. Sem normalização, Playtime
    # dominaria a distância apenas pelo seu magnitude absoluto.

    num_df = df[NUMERIC_COLS].fillna(0.0)
    scaler = MinMaxScaler()
    numeric_scaled = csr_matrix(scaler.fit_transform(num_df))

    # -----------------------------------------------------------------------
    # 3. Aplicar pesos e concatenar
    # -----------------------------------------------------------------------
    # A multiplicação escalar em matriz esparsa é O(nnz) — sem custo de memória.
    # hstack concatena horizontalmente: cada jogo vira 1 vetor com todas features.

    feature_matrix = hstack([
        tags_sparse     * WEIGHTS["tags"],
        genres_sparse   * WEIGHTS["genres"],
        cats_sparse     * WEIGHTS["categories"],
        numeric_scaled  * WEIGHTS["numeric"],
    ]).tocsr()  # CSR = Compressed Sparse Row — mais eficiente para operações de linha

    # -----------------------------------------------------------------------
    # 4. Nomes das features (interpretabilidade)
    # -----------------------------------------------------------------------
    feature_names = (
        [f"tag_{t}" for t in mlb_tags.classes_]
        + [f"genre_{g}" for g in mlb_genres.classes_]
        + [f"cat_{c}" for c in mlb_cats.classes_]
        + [f"num_{c}" for c in NUMERIC_COLS]
    )

    encoders = {
        "mlb_tags": mlb_tags,
        "mlb_genres": mlb_genres,
        "mlb_cats": mlb_cats,
        "scaler": scaler,
    }

    logger.info(
        f"Matriz de features construída: {feature_matrix.shape} "
        f"| Densidade: {feature_matrix.nnz / (feature_matrix.shape[0] * feature_matrix.shape[1]):.2%}"
    )

    return feature_matrix, encoders, feature_names


def get_top_tags(df: pd.DataFrame, n: int = 30) -> pd.Series:
    """Retorna as N tags mais frequentes no dataset — útil para EDA."""
    return (
        df["Tags"]
        .dropna()
        .str.split(",")
        .explode()
        .str.strip()
        .value_counts()
        .head(n)
    )
