"""Baixa os dados públicos do INEP usados no projeto.

Tudo é agregado por escola (o INEP não publica mais dados por aluno desde 2022, por causa da LGPD).
Os arquivos vão para data/raw/ e não entram no git.

Uso: python -m src.coleta
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import certifi
import requests

BASE = "https://download.inep.gov.br"
RAIZ = Path(__file__).resolve().parents[1]
RAW = RAIZ / "data" / "raw"
# O servidor do INEP não envia o certificado intermediário da RNP, então a cadeia TLS fica incompleta.
# Em vez de desligar a verificação, somamos o intermediário oficial (GlobalSign) às CAs confiáveis.
INTERMEDIARIO = RAIZ / "certs" / "rnp-icpedu-gr46.pem"


def bundle_ca() -> str:
    base = os.environ.get("REQUESTS_CA_BUNDLE") or certifi.where()
    tmp = tempfile.NamedTemporaryFile("w", suffix=".pem", delete=False)
    tmp.write(Path(base).read_text() + "\n" + INTERMEDIARIO.read_text())
    tmp.close()
    return tmp.name

# Variáveis da escola no ano t  ->  abandono no ano t+1.
ANOS_FEATURES = [2021, 2022]
ANOS_ALVO = [2022, 2023]


def arquivos() -> dict[str, str]:
    urls: dict[str, str] = {}
    for ano in ANOS_FEATURES:
        urls[f"censo_{ano}.zip"] = f"{BASE}/dados_abertos/microdados_censo_escolar_{ano}.zip"
        for ind in ["TDI", "ATU", "AFD", "IRD", "HAD"]:
            urls[f"{ind.lower()}_{ano}.zip"] = (
                f"{BASE}/informacoes_estatisticas/indicadores_educacionais/{ano}/{ind}_{ano}_ESCOLAS.zip"
            )
    for ano in sorted(set(ANOS_FEATURES + ANOS_ALVO)):
        urls[f"rendimento_{ano}.zip"] = (
            f"{BASE}/informacoes_estatisticas/indicadores_educacionais/{ano}/tx_rend_escolas_{ano}.zip"
        )
    return urls


def baixar(nome: str, url: str, ca: str) -> bool:
    destino = RAW / nome
    if destino.exists() and destino.stat().st_size > 0:
        print(f"ok (já existe) {nome}")
        return True
    with requests.get(url, stream=True, timeout=120, verify=ca) as r:
        if r.status_code != 200 or "zip" not in r.headers.get("Content-Type", "zip"):
            print(f"FALHOU {nome}: HTTP {r.status_code} {r.headers.get('Content-Type')}")
            return False
        tmp = destino.with_suffix(".part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
        tmp.rename(destino)
    print(f"ok {nome} ({destino.stat().st_size / 1e6:.1f} MB)")
    return True


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    ca = bundle_ca()
    falhas = [n for n, u in arquivos().items() if not baixar(n, u, ca)]
    if falhas:
        print("Falharam:", ", ".join(falhas))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
