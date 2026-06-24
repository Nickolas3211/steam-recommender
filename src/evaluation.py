"""
evaluation.py
-------------
Métricas de avaliação para sistemas Content-Based sem histórico de usuários.

O desafio:
    Sistemas Collaborative Filtering podem usar métricas clássicas como
    Precision@K e NDCG porque temos o "ground truth" (o que o usuário realmente
    consumiu). Em Content-Based puro, não temos esse histórico.

Alternativas usadas aqui:
    1. Sanity Check Qualitativo — teste manual com jogos conhecidos
    2. Intra-List Diversity (ILD) — diversidade dentro da lista de recomendações
    3. Catalog Coverage — % do catálogo que aparece como recomendação
    4. Genre Coherence — proporção de recomendações que compartilham pelo menos
       um gênero com o jogo consultado (proxy de relevância temática)
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import logging

logger = logging.getLogger(__name__)


def intra_list_diversity(
    recommendations: pd.DataFrame,
    feature_matrix: csr_matrix,
    rec_indices: list[int],
) -> float:
    """
    Mede a diversidade dentro da lista de recomendações.

    ILD = 1 - (média das similaridades cosseno entre todos os pares na lista)

    Interpretação:
        ILD = 1.0 → recomendações completamente distintas entre si (máxima diversidade)
        ILD = 0.0 → todas as recomendações são idênticas (nenhuma diversidade)

    Um bom sistema equilibra relevância (alta similaridade com a query) e
    diversidade (baixa similaridade entre as próprias recomendações). Listas
    com ILD próximo de 0 sofrem do problema do "filtro bolha".

    Parameters
    ----------
    recommendations : pd.DataFrame
        Saída do método ContentRecommender.recommend().
    feature_matrix : csr_matrix
        Matriz de features completa.
    rec_indices : list[int]
        Índices das recomendações na feature_matrix.

    Returns
    -------
    float
        Score ILD entre 0 e 1.
    """
    if len(rec_indices) < 2:
        return 0.0

    sub_matrix = feature_matrix[rec_indices]
    sim_matrix = cosine_similarity(sub_matrix)

    n = len(rec_indices)
    # Soma dos elementos off-diagonal (pares únicos)
    off_diagonal_sum = sim_matrix.sum() - n  # remove diagonal (self-similarity = 1)
    n_pairs = n * (n - 1)

    avg_similarity = off_diagonal_sum / n_pairs
    return float(1.0 - avg_similarity)


def genre_coherence(query_genres: str, recommendations: pd.DataFrame) -> float:
    """
    Proporção de recomendações que compartilham pelo menos um gênero com a query.

    Serve como proxy de relevância temática — se o usuário gosta de RPG,
    quantas das recomendações também são RPG?

    Parameters
    ----------
    query_genres : str
        String de gêneros do jogo consultado (ex: "Action,RPG,Indie").
    recommendations : pd.DataFrame
        Saída do ContentRecommender.recommend().

    Returns
    -------
    float
        Proporção entre 0.0 e 1.0.
    """
    if not isinstance(query_genres, str) or recommendations.empty:
        return 0.0

    query_set = {g.strip().lower() for g in query_genres.split(",")}

    def has_overlap(genres_str):
        if not isinstance(genres_str, str):
            return False
        rec_set = {g.strip().lower() for g in genres_str.split(",")}
        return bool(query_set & rec_set)

    overlap_count = recommendations["Genres"].apply(has_overlap).sum()
    return float(overlap_count / len(recommendations))


def catalog_coverage(
    recommender,
    sample_games: list[str],
    top_n: int = 10,
) -> dict:
    """
    Calcula a cobertura do catálogo: % de jogos únicos que aparecem
    como recomendação ao longo de múltiplas queries.

    Catálogos com coverage baixo sofrem do "efeito cauda longa" invertido:
    sempre recomendam os mesmos jogos populares, ignorando títulos únicos.

    Parameters
    ----------
    recommender : ContentRecommender
        Instância treinada do recomendador.
    sample_games : list[str]
        Lista de jogos a usar como query de teste.
    top_n : int
        Número de recomendações por query.

    Returns
    -------
    dict com chaves:
        'coverage': float — proporção de jogos únicos recomendados
        'unique_recommendations': int
        'total_catalog': int
        'failed_queries': list[str]
    """
    all_recommended = set()
    failed = []

    for game in sample_games:
        try:
            recs = recommender.recommend(game, top_n=top_n)
            all_recommended.update(recs["game_name"].tolist())
        except ValueError:
            failed.append(game)

    total_catalog = len(recommender.df)
    coverage = len(all_recommended) / total_catalog if total_catalog > 0 else 0.0

    return {
        "coverage": round(coverage, 4),
        "unique_recommendations": len(all_recommended),
        "total_catalog": total_catalog,
        "failed_queries": failed,
    }


def sanity_check(recommender, test_cases: list[tuple[str, list[str]]]) -> pd.DataFrame:
    """
    Executa testes de sanidade qualitativos: verifica se jogos conhecidos
    retornam recomendações esperadas.

    Parameters
    ----------
    recommender : ContentRecommender
        Instância do recomendador.
    test_cases : list of (query, expected_tags)
        Cada tupla contém o nome do jogo e uma lista de tags/gêneros que
        deveriam aparecer nas recomendações (ex: [("Hollow Knight", ["Metroidvania"])]).

    Returns
    -------
    pd.DataFrame
        Resultado de cada teste com score de cobertura de tags esperadas.
    """
    results = []

    for query, expected_tags in test_cases:
        try:
            recs = recommender.recommend(query, top_n=10)
            all_tags_in_recs = (
                recs["Tags"]
                .fillna("")
                .str.lower()
                .str.cat(sep=",")
            )
            tags_found = [t for t in expected_tags if t.lower() in all_tags_in_recs]
            score = len(tags_found) / len(expected_tags) if expected_tags else 0.0

            results.append({
                "query": query,
                "expected_tags": expected_tags,
                "tags_found": tags_found,
                "hit_rate": round(score, 2),
                "top_3_recs": recs["game_name"].head(3).tolist(),
            })
        except ValueError as e:
            results.append({
                "query": query,
                "expected_tags": expected_tags,
                "tags_found": [],
                "hit_rate": None,
                "top_3_recs": [str(e)],
            })

    return pd.DataFrame(results)


def full_evaluation_report(recommender, feature_matrix: csr_matrix) -> None:
    """
    Executa o conjunto completo de avaliações e imprime um relatório formatado.
    """
    print("=" * 60)
    print("   RELATÓRIO DE AVALIAÇÃO — SISTEMA DE RECOMENDAÇÃO")
    print("=" * 60)

    # --- Sanity Check ---
    print("\n📋 SANITY CHECK QUALITATIVO")
    print("-" * 40)
    test_cases = [
        ("Hollow Knight", ["Metroidvania", "Souls-like", "Platformer", "Indie"]),
        ("Terraria",      ["Sandbox", "Crafting", "Open World", "Survival"]),
        ("Stardew Valley",["Farming Sim", "Relaxing", "Life Sim", "RPG"]),
        ("Hades",         ["Action Roguelike", "Rogue-lite", "Hack and Slash"]),
        ("Celeste",       ["Precision Platformer", "Difficult", "Platformer"]),
    ]

    sanity_df = sanity_check(recommender, test_cases)
    for _, row in sanity_df.iterrows():
        status = "✅" if (row["hit_rate"] or 0) >= 0.5 else "⚠️"
        print(f"\n{status} Query: '{row['query']}'")
        print(f"   Hit Rate: {row['hit_rate']} | Tags encontradas: {row['tags_found']}")
        print(f"   Top 3 recomendações: {row['top_3_recs']}")

    # --- ILD e Genre Coherence ---
    print("\n\n📊 MÉTRICAS POR JOGO")
    print("-" * 40)

    for query, _ in test_cases:
        try:
            game_info = recommender.get_game_info(query)
            recs = recommender.recommend(query, top_n=10)
            rec_indices = recommender.df[
                recommender.df["game_name"].isin(recs["game_name"])
            ].index.tolist()[:10]

            ild = intra_list_diversity(recs, feature_matrix, rec_indices)
            gc = genre_coherence(game_info.get("Genres", ""), recs)
            avg_sim = recs["similarity_score"].mean()

            print(f"\n🎮 {query}")
            print(f"   Similaridade média: {avg_sim:.3f}")
            print(f"   ILD (diversidade): {ild:.3f}")
            print(f"   Genre Coherence:   {gc:.2%}")
        except Exception as e:
            print(f"   Erro para '{query}': {e}")

    # --- Catalog Coverage ---
    print("\n\n🗂️  CATALOG COVERAGE")
    print("-" * 40)
    sample_games = [q for q, _ in test_cases]
    cov = catalog_coverage(recommender, sample_games, top_n=10)
    print(f"   Jogos únicos recomendados: {cov['unique_recommendations']}")
    print(f"   Total do catálogo:         {cov['total_catalog']:,}")
    print(f"   Coverage:                  {cov['coverage']:.2%}")
    if cov["failed_queries"]:
        print(f"   ⚠️  Queries com falha: {cov['failed_queries']}")

    print("\n" + "=" * 60)
