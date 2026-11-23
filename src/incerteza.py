"""Intervalos de confiança (bootstrap por escola) para a diferença entre o modelo e a regra simples.

Uso: python -m src.incerteza
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.modelo import REL, BASE, enriquecer, pipelines, preparar_xy, captura_no_topo


def main(n_boot: int = 1000) -> None:
    df = enriquecer(pd.read_parquet(BASE))
    tr, te = df[df["ano"] == 2021], df[df["ano"] == 2022]
    x_tr, y_tr = preparar_xy(tr)
    x_te, y_te = preparar_xy(te)
    gb = pipelines()["gradient_boosting"].fit(x_tr, y_tr)
    s_gb = gb.predict_proba(x_te)[:, 1]
    s_p = te["abandono_med_rank"].fillna(0.5).to_numpy()
    y = y_te.to_numpy()
    rng = np.random.default_rng(42)
    saida = {}
    for recorte, mask in {"nordeste": np.ones(len(y), bool), "rn": (te["SG_UF"] == "RN").to_numpy()}.items():
        idx_all = np.flatnonzero(mask)
        difs = {"roc_auc": [], "pr_auc": [], "captura_top20": []}
        for _ in range(n_boot):
            i = rng.choice(idx_all, size=len(idx_all), replace=True)
            if y[i].min() == y[i].max():
                continue
            difs["roc_auc"].append(roc_auc_score(y[i], s_gb[i]) - roc_auc_score(y[i], s_p[i]))
            difs["pr_auc"].append(average_precision_score(y[i], s_gb[i]) - average_precision_score(y[i], s_p[i]))
            difs["captura_top20"].append(captura_no_topo(y[i], s_gb[i]) - captura_no_topo(y[i], s_p[i]))
        saida[recorte] = {k: {"dif": round(float(np.mean(v)), 3),
                              "ic95": [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]}
                          for k, v in difs.items()}
    (REL / "incerteza.json").write_text(json.dumps(saida, ensure_ascii=False, indent=2))
    print(json.dumps(saida, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
