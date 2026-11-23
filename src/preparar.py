"""Monta a base analítica: uma linha por escola e ano.

Recorte: escolas estaduais do Nordeste em funcionamento, com ensino médio (>= 30 matrículas).
Variáveis medidas no ano t; alvo = taxa de abandono do ensino médio no ano t+1.

Uso: python -m src.preparar
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
RAW = RAIZ / "data" / "raw"
OUT = RAIZ / "data" / "processed"

UFS_NORDESTE = ["MA", "PI", "CE", "RN", "PB", "PE", "AL", "SE", "BA"]
MIN_MAT_MED = 30

CENSO_COLS = [
    "CO_ENTIDADE", "SG_UF", "CO_MUNICIPIO", "NO_MUNICIPIO", "NO_ENTIDADE", "TP_DEPENDENCIA",
    "TP_SITUACAO_FUNCIONAMENTO", "TP_LOCALIZACAO", "TP_LOCALIZACAO_DIFERENCIADA", "IN_MED", "IN_PROF_TEC",
    "IN_NOTURNO", "IN_BIBLIOTECA", "IN_LABORATORIO_CIENCIAS", "IN_LABORATORIO_INFORMATICA", "IN_INTERNET_ALUNOS",
    "IN_BANDA_LARGA", "IN_QUADRA_ESPORTES", "IN_REFEITORIO", "IN_AGUA_POTAVEL", "IN_ESGOTO_REDE_PUBLICA",
    "IN_ORGAO_GREMIO_ESTUDANTIL", "IN_EXAME_SELECAO", "QT_SALAS_UTILIZADAS", "QT_PROF_PSICOLOGO",
    "QT_PROF_ASSIST_SOCIAL", "QT_MAT_BAS", "QT_MAT_MED", "QT_MAT_MED_INT", "QT_MAT_PROF_TEC", "QT_MAT_BAS_N",
    "QT_MAT_BAS_18_MAIS", "QT_MAT_BAS_15_17", "QT_MAT_BAS_PRETA", "QT_MAT_BAS_PARDA", "QT_MAT_BAS_FEM",
    "QT_DOC_MED", "QT_TUR_MED",
]


def _csv_do_zip(caminho: Path) -> pd.DataFrame:
    with zipfile.ZipFile(caminho) as z:
        nome = next(n for n in z.namelist() if n.lower().endswith(".csv") and "/dados/" in n.lower())
        with z.open(nome) as f:
            return pd.read_csv(io.TextIOWrapper(f, encoding="latin-1"), sep=";", usecols=CENSO_COLS, low_memory=False)


def _xlsx_do_zip(caminho: Path) -> pd.DataFrame:
    """Planilhas de indicadores do INEP: acha a linha de códigos (NU_ANO_CENSO) e usa como cabeçalho."""
    with zipfile.ZipFile(caminho) as z:
        nome = next(n for n in z.namelist() if n.lower().endswith(".xlsx"))
        dados = z.read(nome)
    topo = pd.read_excel(io.BytesIO(dados), engine="calamine", header=None, nrows=20)
    linha = topo.index[topo.iloc[:, 0].astype(str).str.strip() == "NU_ANO_CENSO"][0]
    df = pd.read_excel(io.BytesIO(dados), engine="calamine", header=linha)
    df = df[df["SG_UF"].isin(UFS_NORDESTE)]
    return df


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.replace({"--": np.nan, "---": np.nan}), errors="coerce")


def indicadores(ano: int) -> pd.DataFrame:
    rend = _xlsx_do_zip(RAW / f"rendimento_{ano}.zip")
    base = pd.DataFrame({
        "CO_ENTIDADE": rend["CO_ENTIDADE"].astype("int64"),
        "aprovacao_med": _num(rend["1_CAT_MED"]),
        "reprovacao_med": _num(rend["2_CAT_MED"]),
        "abandono_med": _num(rend["3_CAT_MED"]),
    })
    extras = {"tdi": ("TDI", "MED_CAT_0"), "atu": ("ATU", "MED_CAT_0"), "had": ("HAD", "MED_CAT_0"),
              "afd": ("AFD", "MED_CAT_1"), "ird": ("IRD", "EDU_BAS_CAT_0")}
    for nome, (_, coluna) in extras.items():
        caminho = RAW / f"{nome}_{ano}.zip"
        if not caminho.exists():
            continue
        df = _xlsx_do_zip(caminho)
        col = f"{nome}_med" if nome != "ird" else "ird"
        if nome == "afd":
            col = "afd_grupo1_med"
        base = base.merge(
            pd.DataFrame({"CO_ENTIDADE": df["CO_ENTIDADE"].astype("int64"), col: _num(df[coluna])}),
            on="CO_ENTIDADE", how="left",
        )
    return base


def censo(ano: int) -> pd.DataFrame:
    df = _csv_do_zip(RAW / f"censo_{ano}.zip")
    df = df[
        df["SG_UF"].isin(UFS_NORDESTE)
        & (df["TP_DEPENDENCIA"] == 2)  # estadual
        & (df["TP_SITUACAO_FUNCIONAMENTO"] == 1)
        & (df["IN_MED"] == 1)
        & (df["QT_MAT_MED"] >= MIN_MAT_MED)
    ].copy()
    mat = df["QT_MAT_BAS"].replace(0, np.nan)
    df["rural"] = (df["TP_LOCALIZACAO"] == 2).astype(int)
    df["localizacao_diferenciada"] = (df["TP_LOCALIZACAO_DIFERENCIADA"].fillna(0) > 0).astype(int)
    df["oferta_tecnica"] = df["IN_PROF_TEC"].fillna(0).astype(int)
    df["pct_med_integral"] = df["QT_MAT_MED_INT"] / df["QT_MAT_MED"]
    df["pct_mat_noturno"] = df["QT_MAT_BAS_N"] / mat
    df["pct_18_mais"] = df["QT_MAT_BAS_18_MAIS"] / mat
    df["pct_preta_parda"] = (df["QT_MAT_BAS_PRETA"] + df["QT_MAT_BAS_PARDA"]) / mat
    df["pct_feminino"] = df["QT_MAT_BAS_FEM"] / mat
    df["alunos_por_docente_med"] = df["QT_MAT_MED"] / df["QT_DOC_MED"].replace(0, np.nan)
    df["tem_psicologo_ou_as"] = ((df["QT_PROF_PSICOLOGO"].fillna(0) + df["QT_PROF_ASSIST_SOCIAL"].fillna(0)) > 0).astype(int)
    df["log_mat_med"] = np.log1p(df["QT_MAT_MED"])
    return df


def montar(ano: int) -> pd.DataFrame:
    c = censo(ano)
    ind = indicadores(ano)
    alvo = indicadores(ano + 1)[["CO_ENTIDADE", "abandono_med"]].rename(columns={"abandono_med": "abandono_med_t1"})
    df = c.merge(ind, on="CO_ENTIDADE", how="left").merge(alvo, on="CO_ENTIDADE", how="inner")
    df["ano"] = ano
    return df[df["abandono_med_t1"].notna()]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    partes = []
    for ano in (2021, 2022):
        df = montar(ano)
        print(ano, df.shape, "escolas RN:", (df["SG_UF"] == "RN").sum())
        partes.append(df)
    base = pd.concat(partes, ignore_index=True)
    base.to_parquet(OUT / "base_escolas.parquet", index=False)
    print("salvo", OUT / "base_escolas.parquet", base.shape)


if __name__ == "__main__":
    main()
