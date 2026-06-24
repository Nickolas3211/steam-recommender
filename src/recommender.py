"""
recommender.py
--------------
Motor de recomendação Content-Based usando Similaridade do Cosseno.

Por que Cosseno e não Distância Euclidiana?
    Jogos populares tendem a ter mais tags que jogos indie (vetores de magnitude
    maior). A distância Euclidiana seria influenciada por essa magnitude, criando
    viés contra títulos indie. O Cosseno mede apenas o ÂNGULO entre vetores,
    ignorando magnitude — dois jogos com as mesmas proporções de tags terão
    cosseno = 1.0 independentemente de quantas tags têm ao total.

    Cosseno(A, B) = (A · B) / (||A|| × ||B||)  ∈ [-1, 1]
    Para dados binários (0/1): resultado sempre em [0, 1].

Estratégia on-demand:
    Em vez de pré-calcular a matriz N×N completa (56k × 56k = 3 bilhões de células),
    calculamos a similaridade apenas para o jogo consultado no momento da query.
    Isso reduz o custo de memória de ~24 GB para ~4 MB por consulta.
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)


class ContentRecommender:
    """
    Sistema de recomendação Content-Based usando Similaridade do Cosseno.

    Attributes
    ----------
    df : pd.DataFrame
        Dataset limpo com informações dos jogos.
    feature_matrix : csr_matrix
        Matriz esparsa de features (n_jogos × n_features).
    _name_to_idx : dict
        Mapeamento rápido nome_do_jogo → índice da linha na feature_matrix.
    """

    def __init__(self, df: pd.DataFrame, feature_matrix: csr_matrix):
        self.df = df.copy()
        self.feature_matrix = feature_matrix

        # Índice invertido para busca O(1) por nome
        self._name_to_idx: dict[str, list[int]] = {}
        for idx, name in enumerate(df["game_name"]):
            if isinstance(name, str):
                self._name_to_idx.setdefault(name.lower(), []).append(idx)

        logger.info(
            f"ContentRecommender inicializado: {len(df):,} jogos | "
            f"Feature matrix: {feature_matrix.shape}"
        )

    # -----------------------------------------------------------------------
    # API pública
    # -----------------------------------------------------------------------

    def recommend(
        self,
        game_name: str,
        top_n: int = 10,
        exclude_same_developer: bool = False,
        min_reviews: int = 0,
    ) -> pd.DataFrame:
        """
        Retorna os top_n jogos mais similares ao jogo informado.

        Parameters
        ----------
        game_name : str
            Nome do jogo (busca case-insensitive, correspondência parcial aceita).
        top_n : int
            Número de recomendações a retornar.
        exclude_same_developer : bool
            Se True, exclui jogos do mesmo desenvolvedor (evita auto-promoção).
        min_reviews : int
            Filtra recomendações com menos de X avaliações totais.

        Returns
        -------
        pd.DataFrame
            DataFrame com colunas: game_name, Genres, Tags, review_score,
            Price, total_reviews, similarity_score.

        Raises
        ------
        ValueError
            Se o jogo não for encontrado no dataset.
        """
        idx = self._resolve_game(game_name)
        query_vector = self.feature_matrix[idx]  # shape: (1, n_features)

        # Calcular similaridade do jogo query contra todos os outros
        # cosine_similarity retorna array shape (1, n_jogos)
        scores = cosine_similarity(query_vector, self.feature_matrix).flatten()

        # Criar série indexada para facilitar filtragens
        score_series = pd.Series(scores, index=self.df.index, name="similarity_score")

        # Excluir o próprio jogo (similarity = 1.0 com ele mesmo)
        score_series = score_series.drop(index=idx)

        # Filtros opcionais
        if min_reviews > 0:
            valid = self.df["total_reviews"] >= min_reviews
            score_series = score_series[valid.drop(index=idx).values]

        if exclude_same_developer and "Developers" in self.df.columns:
            dev = self.df.loc[idx, "Developers"]
            if pd.notna(dev) and dev:
                same_dev = self.df["Developers"].str.contains(
                    str(dev), na=False, regex=False
                )
                score_series = score_series[~same_dev.drop(index=idx).values]

        # Top-N por score
        top_indices = score_series.nlargest(top_n).index

        # Montar resultado
        output_cols = ["steam_id", "game_name", "Genres", "Tags", "review_score",
                       "Price", "total_reviews", "Header image", "Metacritic score", "Average playtime forever", "Developers"]
        available = [c for c in output_cols if c in self.df.columns]
        result = self.df.loc[top_indices, available].copy()
        result["similarity_score"] = score_series[top_indices].values.round(4)

        return result.reset_index(drop=True)

    def get_game_info(self, game_name: str) -> pd.Series:
        """Retorna as informações do jogo consultado."""
        idx = self._resolve_game(game_name)
        return self.df.loc[idx]

    def search(self, partial_name: str, top_k: int = 5) -> list[str]:
        """
        Busca por correspondência parcial de nome (case-insensitive).
        Útil quando o usuário não sabe o nome exato do jogo.
        """
        partial = partial_name.lower()
        matches = [
            name for name in self.df["game_name"].dropna()
            if partial in name.lower()
        ]
        return matches[:top_k]

    # -----------------------------------------------------------------------
    # Helpers privados
    # -----------------------------------------------------------------------

    def _resolve_game(self, game_name: str) -> int:
        """
        Resolve o nome do jogo para um índice numérico.
        Tenta correspondência exata primeiro, depois parcial.
        """
        key = game_name.lower().strip()

        # Correspondência exata
        if key in self._name_to_idx:
            indices = self._name_to_idx[key]
            if len(indices) > 1:
                logger.warning(
                    f"'{game_name}' tem {len(indices)} entradas. Usando a primeira."
                )
            return indices[0]

        # Correspondência parcial
        partial_matches = [
            (name, idxs[0])
            for name, idxs in self._name_to_idx.items()
            if key in name
        ]

        if not partial_matches:
            suggestions = self.search(game_name, top_k=5)
            raise ValueError(
                f"Jogo '{game_name}' não encontrado.\n"
                f"Sugestões: {suggestions}"
            )

        if len(partial_matches) > 1:
            names = [m[0] for m in partial_matches[:5]]
            logger.warning(
                f"Múltiplas correspondências para '{game_name}': {names}. "
                f"Usando '{partial_matches[0][0]}'."
            )

        return partial_matches[0][1]
