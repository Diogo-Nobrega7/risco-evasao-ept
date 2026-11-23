"""Treina e avalia o modelo de risco de abandono alto no ensino médio.

Validação temporal: treina com 2021 -> 2022 e testa com 2022 -> 2023 (o futuro nunca entra no treino).
Alvo: a escola estar entre as 25% de maior abandono no ensino médio do seu estado, no ano seguinte.
O nível de abandono muda muito entre anos (pós-pandemia) e entre estados; um limiar fixo misturaria
essas mudanças com o risco da escola. O alvo relativo responde à pergunta da gestão: quais escolas da
minha rede priorizar?

Uso: python -m src.modelo
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RAIZ = Path(__file__).resolve().parents[1]
BASE = RAIZ / "data" / "processed" / "base_escolas.parquet"
REL = RAIZ / "relatorio"
QUARTIL = 0.75  # escola acima deste percentil do estado no ano seguinte = alvo positivo
RELATIVAS = ["abandono_med", "reprovacao_med", "aprovacao_med", "tdi_med"]
TOP = 0.20  # fração de escolas que uma equipe consegue priorizar

NUMERICAS = [
    "abandono_med_rank", "reprovacao_med_rank", "aprovacao_med_rank", "tdi_med_rank",
    "abandono_med", "reprovacao_med", "aprovacao_med", "tdi_med", "atu_med", "had_med", "afd_grupo1_med", "ird",
    "pct_med_integral", "pct_mat_noturno", "pct_18_mais", "pct_preta_parda", "pct_feminino", "alunos_por_docente_med",
    "log_mat_med", "QT_SALAS_UTILIZADAS",
]
BINARIAS = [
    "rural", "localizacao_diferenciada", "oferta_tecnica", "IN_NOTURNO", "IN_BIBLIOTECA", "IN_LABORATORIO_CIENCIAS",
    "IN_LABORATORIO_INFORMATICA", "IN_INTERNET_ALUNOS", "IN_BANDA_LARGA", "IN_QUADRA_ESPORTES", "IN_REFEITORIO",
    "IN_AGUA_POTAVEL", "IN_ESGOTO_REDE_PUBLICA", "IN_ORGAO_GREMIO_ESTUDANTIL", "IN_EXAME_SELECAO", "tem_psicologo_ou_as",
]
CATEGORICAS = ["SG_UF"]
FEATURES = NUMERICAS + BINARIAS + CATEGORICAS

NOMES = {
    "abandono_med_rank": "Abandono no EM (posição no estado)", "reprovacao_med_rank": "Reprovação no EM (posição no estado)",
    "aprovacao_med_rank": "Aprovação no EM (posição no estado)", "tdi_med_rank": "Distorção idade-série (posição no estado)",
    "abandono_med": "Abandono no EM (ano atual)", "reprovacao_med": "Reprovação no EM", "aprovacao_med": "Aprovação no EM",
    "tdi_med": "Distorção idade-série no EM", "atu_med": "Alunos por turma no EM", "had_med": "Horas-aula diárias no EM",
    "afd_grupo1_med": "Docentes com formação adequada", "ird": "Regularidade do corpo docente",
    "pct_med_integral": "% do EM em tempo integral", "pct_mat_noturno": "% de matrículas no noturno",
    "pct_18_mais": "% de alunos com 18 anos ou mais", "pct_preta_parda": "% de alunos pretos e pardos",
    "pct_feminino": "% de alunas", "alunos_por_docente_med": "Alunos por docente no EM", "log_mat_med": "Tamanho do EM",
    "QT_SALAS_UTILIZADAS": "Salas em uso", "rural": "Escola rural", "oferta_tecnica": "Oferta de curso técnico",
    "IN_NOTURNO": "Funciona à noite", "SG_UF": "Estado",
}


def enriquecer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    grupo = df.groupby(["ano", "SG_UF"])
    for c in RELATIVAS:
        df[f"{c}_rank"] = grupo[c].rank(pct=True)
    pos = grupo["abandono_med_t1"].rank(pct=True)
    df["alvo"] = ((pos > QUARTIL) & (df["abandono_med_t1"] > 0)).astype(int)
    return df


def preparar_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    x = df[FEATURES].copy()
    for c in BINARIAS:
        x[c] = x[c].fillna(0).astype(float)
    return x, df["alvo"]


def captura_no_topo(y: np.ndarray, score: np.ndarray, frac: float = TOP) -> float:
    """Das escolas com abandono alto, quantas % estão entre as `frac` de maior risco."""
    n = max(1, int(round(len(score) * frac)))
    topo = np.argsort(-score)[:n]
    return float(y[topo].sum() / max(1, y.sum()))


def metricas(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": round(float(roc_auc_score(y, score)), 3),
        "pr_auc": round(float(average_precision_score(y, score)), 3),
        "captura_top20": round(captura_no_topo(y, score), 3),
        "taxa_base": round(float(y.mean()), 3),
        "n": int(len(y)),
    }


def pipelines() -> dict[str, Pipeline]:
    pre_lin = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("esc", StandardScaler())]), NUMERICAS),
        ("bin", "passthrough", BINARIAS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAS),
    ])
    pre_arv = ColumnTransformer([
        ("num", "passthrough", NUMERICAS + BINARIAS),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAS),
    ])
    return {
        "regressao_logistica": Pipeline([("pre", pre_lin), ("clf", LogisticRegression(max_iter=2000, C=0.5))]),
        "gradient_boosting": Pipeline([
            ("pre", pre_arv),
            ("clf", HistGradientBoostingClassifier(
                learning_rate=0.05, max_iter=400, max_leaf_nodes=15, min_samples_leaf=40,
                l2_regularization=1.0, early_stopping=True, validation_fraction=0.15, random_state=42,
            )),
        ]),
    }


def grafico_importancia(imp: pd.DataFrame, caminho: Path) -> None:
    top = imp.head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)
    ax.barh([NOMES.get(f, f) for f in top["variavel"]], top["queda_pr_auc"], color="#10b981")
    ax.set_xlabel("Queda na PR-AUC quando a variável é embaralhada")
    ax.set_title("O que mais pesa no risco (teste 2022 → 2023)", loc="left", fontsize=11)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(caminho)
    plt.close(fig)


def grafico_captura(y: np.ndarray, scores: dict[str, np.ndarray], caminho: Path) -> None:
    fracs = np.linspace(0.01, 1, 100)
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    cores = {"gradient_boosting": "#10b981", "regressao_logistica": "#3b82f6", "persistencia": "#a1a1aa"}
    rotulos = {"gradient_boosting": "Gradient boosting", "regressao_logistica": "Regressão logística",
               "persistencia": "Só o abandono do ano atual"}
    for nome, s in scores.items():
        ax.plot(fracs * 100, [captura_no_topo(y, s, f) * 100 for f in fracs], label=rotulos[nome], color=cores[nome], lw=2)
    ax.plot([0, 100], [0, 100], ls="--", color="#d4d4d8", lw=1, label="Escolha aleatória")
    ax.axvline(TOP * 100, color="#71717a", lw=0.8, ls=":")
    ax.set_xlabel("% das escolas priorizadas (por ordem de risco)")
    ax.set_ylabel("% das escolas de maior abandono encontradas")
    ax.set_title("Quantas escolas de maior abandono a priorização encontra", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(caminho)
    plt.close(fig)


def main() -> None:
    REL.mkdir(exist_ok=True)
    df = enriquecer(pd.read_parquet(BASE))
    treino, teste = df[df["ano"] == 2021], df[df["ano"] == 2022]
    x_tr, y_tr = preparar_xy(treino)
    x_te, y_te = preparar_xy(teste)

    resultados: dict[str, dict] = {}
    scores: dict[str, np.ndarray] = {}

    persist = teste["abandono_med_rank"].fillna(0.5).to_numpy()
    scores["persistencia"] = persist
    resultados["persistencia"] = {"nordeste": metricas(y_te.to_numpy(), persist)}

    modelos = pipelines()
    for nome, pipe in modelos.items():
        pipe.fit(x_tr, y_tr)
        s = pipe.predict_proba(x_te)[:, 1]
        scores[nome] = s
        resultados[nome] = {"nordeste": metricas(y_te.to_numpy(), s)}

    rn = (teste["SG_UF"] == "RN").to_numpy()
    for nome, s in scores.items():
        resultados[nome]["rn"] = metricas(y_te.to_numpy()[rn], s[rn])
        tec = (teste["oferta_tecnica"] == 1).to_numpy()
        resultados[nome]["escolas_com_tecnico"] = metricas(y_te.to_numpy()[tec], s[tec])

    gb = modelos["gradient_boosting"]
    pi = permutation_importance(gb, x_te, y_te, scoring="average_precision", n_repeats=10, random_state=42, n_jobs=-1)
    imp = pd.DataFrame({"variavel": FEATURES, "queda_pr_auc": pi.importances_mean, "dp": pi.importances_std})
    imp = imp.sort_values("queda_pr_auc", ascending=False)
    imp.round(4).to_csv(REL / "importancia_variaveis.csv", index=False)

    grafico_importancia(imp, REL / "importancia.png")
    grafico_captura(y_te.to_numpy(), scores, REL / "captura.png")

    # Risco agregado por município no RN (não publicamos ranking de escolas com nome).
    rn_df = teste[rn].assign(risco=scores["gradient_boosting"][rn], alto=y_te.to_numpy()[rn])
    por_mun = (rn_df.groupby("NO_MUNICIPIO")
               .agg(escolas=("CO_ENTIDADE", "size"), risco_medio=("risco", "mean"), escolas_no_quartil_superior_2023=("alto", "sum"))
               .sort_values("risco_medio", ascending=False).round(3))
    por_mun.to_csv(REL / "rn_risco_por_municipio.csv")

    resumo = {
        "alvo": "escola entre as 25% de maior abandono no EM do seu estado, no ano seguinte",
        "treino": "2021 -> 2022", "teste": "2022 -> 2023",
        "n_treino": int(len(treino)), "n_teste": int(len(teste)),
        "resultados": resultados,
        "top_variaveis": imp.head(8)["variavel"].map(lambda v: NOMES.get(v, v)).tolist(),
    }
    (REL / "metricas.json").write_text(json.dumps(resumo, ensure_ascii=False, indent=2))
    print(json.dumps(resumo, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
