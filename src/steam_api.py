"""
steam_api.py
------------
Busca dados em tempo real da Steam Web API (gratuita, sem autenticação).

Endpoints utilizados:
    1. Store API  → preço atual, gêneros, desenvolvedores, data de lançamento
       https://store.steampowered.com/api/appdetails?appids={ID}

    2. Reviews API → score e contagem de avaliações em tempo real
       https://store.steampowered.com/appreviews/{ID}?json=1

Por que apenas para o jogo consultado e não para os 56k do catálogo?
    - Rate limit da Steam: ~200 requests/5 min por IP
    - Buscar todos os jogos levaria horas e violaria os ToS para uso em escala
    - O valor para o usuário está em ver o dado ATUAL do jogo que ele perguntou,
      não nas recomendações (que são baseadas em conteúdo, não em preço)

Estrutura da resposta (appdetails):
    {
      "APPID": {
        "success": true,
        "data": {
          "name": "Hollow Knight",
          "price_overview": {
            "final": 3999,          ← centavos (USD)
            "final_formatted": "$9.99",
            "discount_percent": 0
          },
          "metacritic": { "score": 87 },
          "genres": [{"description": "Action"}, ...],
          "developers": ["Team Cherry"],
          "release_date": { "date": "11 Feb, 2017" }
        }
      }
    }

Estrutura da resposta (appreviews):
    {
      "query_summary": {
        "total_positive": 154832,
        "total_reviews": 163510,
        "review_score": 9,
        "review_score_desc": "Overwhelmingly Positive"
      }
    }
"""

import requests
import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Timeout conservador para não travar a UI
REQUEST_TIMEOUT = 8  # segundos

# Mapeamento do score numérico da Steam para label em português
SCORE_LABELS_PT = {
    9: "Extremamente Positivo",
    8: "Muito Positivo",
    7: "Positivo",
    6: "Principalmente Positivo",
    5: "Misto",
    4: "Principalmente Negativo",
    3: "Negativo",
    2: "Muito Negativo",
    1: "Extremamente Negativo",
}


def _get(url: str, params: dict = None) -> Optional[dict]:
    """Wrapper de request com tratamento de erro robusto."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; SteamRecommender/1.0)",
            "Accept-Language": "en-US,en;q=0.9",
        }
        r = requests.get(url, params=params, headers=headers,
                         timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.Timeout:
        logger.warning(f"Timeout ao acessar: {url}")
    except requests.exceptions.HTTPError as e:
        logger.warning(f"HTTP error {e.response.status_code}: {url}")
    except requests.exceptions.ConnectionError:
        logger.warning(f"Sem conexão: {url}")
    except Exception as e:
        logger.warning(f"Erro inesperado em {url}: {e}")
    return None


def fetch_game_details(app_id: int) -> dict:
    """
    Busca detalhes completos de um jogo na Steam Store API.

    Parameters
    ----------
    app_id : int
        Steam AppID do jogo (ex: 367520 para Hollow Knight).

    Returns
    -------
    dict com chaves:
        name, price_usd, price_formatted, discount_pct, is_free,
        metacritic, genres, developers, release_date, success
    """
    url = "https://store.steampowered.com/api/appdetails"
    data = _get(url, params={"appids": app_id, "cc": "us", "l": "english"})

    result = {
        "success": False,
        "name": None,
        "price_usd": None,
        "price_formatted": None,
        "discount_pct": 0,
        "is_free": False,
        "metacritic": None,
        "genres": [],
        "developers": [],
        "release_date": None,
    }

    if not data:
        return result

    app_data = data.get(str(app_id), {})
    if not app_data.get("success"):
        return result

    d = app_data.get("data", {})
    result["success"] = True
    result["name"] = d.get("name")
    result["is_free"] = d.get("is_free", False)
    result["developers"] = d.get("developers", [])
    result["release_date"] = d.get("release_date", {}).get("date")
    result["metacritic"] = d.get("metacritic", {}).get("score")
    result["genres"] = [g["description"] for g in d.get("genres", [])]

    price = d.get("price_overview", {})
    if price:
        # Steam retorna o preço em centavos (3999 = $9.99 USD * 400 = BRL?)
        # final_formatted já vem formatado corretamente
        result["price_formatted"] = price.get("final_formatted", "N/A")
        result["price_usd"] = price.get("final", 0) / 100
        result["discount_pct"] = price.get("discount_percent", 0)
    elif result["is_free"]:
        result["price_formatted"] = "Grátis"
        result["price_usd"] = 0.0

    return result


def fetch_review_summary(app_id: int) -> dict:
    """
    Busca o resumo de avaliações de um jogo (todas as línguas, todas as compras).

    Parameters
    ----------
    app_id : int
        Steam AppID do jogo.

    Returns
    -------
    dict com chaves:
        total_positive, total_negative, total_reviews,
        score (0-9), score_label_en, score_label_pt,
        positive_pct, success
    """
    url = f"https://store.steampowered.com/appreviews/{app_id}"
    data = _get(url, params={
        "json": 1,
        "language": "all",
        "purchase_type": "all",
        "num_per_page": 0,  # só queremos o summary, não reviews individuais
    })

    result = {
        "success": False,
        "total_positive": None,
        "total_negative": None,
        "total_reviews": None,
        "score": None,
        "score_label_en": None,
        "score_label_pt": None,
        "positive_pct": None,
    }

    if not data:
        return result

    summary = data.get("query_summary", {})
    if not summary:
        return result

    total_pos = summary.get("total_positive", 0)
    total_rev = summary.get("total_reviews", 0)

    result["success"] = True
    result["total_positive"] = total_pos
    result["total_negative"] = summary.get("total_negative", 0)
    result["total_reviews"] = total_rev
    result["score"] = summary.get("review_score")
    result["score_label_en"] = summary.get("review_score_desc", "")
    result["score_label_pt"] = SCORE_LABELS_PT.get(
        summary.get("review_score", 0), "Sem avaliações suficientes"
    )
    result["positive_pct"] = (
        round(total_pos / total_rev, 4) if total_rev > 0 else 0.0
    )

    return result


def fetch_live_data(app_id: int) -> dict:
    """
    Busca detalhes + reviews em sequência para um único jogo.
    Retorna um dicionário consolidado pronto para exibição na UI.

    Parameters
    ----------
    app_id : int
        Steam AppID.

    Returns
    -------
    dict consolidado com todos os campos ao vivo, mais 'any_success' bool.
    """
    details = fetch_game_details(app_id)
    # Pequena pausa para não disparar rate limit caso chamado em sequência
    time.sleep(0.3)
    reviews = fetch_review_summary(app_id)

    return {
        # Metadados
        "any_success": details["success"] or reviews["success"],
        "details_ok": details["success"],
        "reviews_ok": reviews["success"],

        # Dados da Store
        "name": details.get("name"),
        "price_formatted": details.get("price_formatted"),
        "price_usd": details.get("price_usd"),
        "discount_pct": details.get("discount_pct", 0),
        "is_free": details.get("is_free", False),
        "metacritic": details.get("metacritic"),
        "genres_live": details.get("genres", []),
        "developers_live": details.get("developers", []),
        "release_date_live": details.get("release_date"),

        # Dados de Reviews
        "total_positive": reviews.get("total_positive"),
        "total_reviews": reviews.get("total_reviews"),
        "positive_pct": reviews.get("positive_pct"),
        "score_label_pt": reviews.get("score_label_pt"),
        "score_label_en": reviews.get("score_label_en"),
    }
