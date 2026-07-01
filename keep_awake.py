"""
Keep-alive para app Streamlit Community Cloud.

O que este script faz:
1. Abre a URL do app num navegador headless (Playwright).
2. Se o app estiver dormindo, a Streamlit Cloud mostra uma tela com o
   botão "Yes, get this app back up!" — o script procura esse botão e
   clica nele.
3. Se o app já estiver ativo, o script apenas confirma isso e termina
   sem fazer nada (visita "inofensiva").

Por que Playwright (navegador real) e não apenas requests/curl:
a reativação do app depende de um clique real na página (evento JS), uma
simples requisição HTTP GET não é suficiente para acordar o app de forma
confiável.
"""

import sys
from playwright.sync_api import sync_playwright

STREAMLIT_URL = "https://nickolas3211-steam-recommender.streamlit.app"  # TODO: ajuste se a URL mudar
WAKE_BUTTON_TEXT = "Yes, get this app back up!"
TIMEOUT_MS = 30_000


def keep_app_awake(url: str) -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        try:
            page.goto(url, timeout=TIMEOUT_MS, wait_until="domcontentloaded")

            # Streamlit Cloud carrega a tela de "sleeping app" de forma assíncrona;
            # damos um tempo curto para o botão aparecer, se for o caso.
            wake_button = page.get_by_text(WAKE_BUTTON_TEXT, exact=False)
            try:
                wake_button.wait_for(timeout=8_000)
                wake_button.click()
                print("App estava dormindo — clique de reativação enviado.")
                # Espera o app efetivamente subir antes de fechar o navegador
                page.wait_for_timeout(15_000)
            except Exception:
                print("Nenhum botão de reativação encontrado — app já estava ativo.")

        except Exception as e:
            print(f"Erro ao acessar o app: {e}")
            sys.exit(1)
        finally:
            browser.close()


if __name__ == "__main__":
    keep_app_awake(STREAMLIT_URL)
