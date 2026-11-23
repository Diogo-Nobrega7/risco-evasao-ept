import numpy as np
import pandas as pd

from src.modelo import captura_no_topo, enriquecer


def test_captura_no_topo_ordena_por_risco():
    y = np.array([1, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    score = np.array([0.9, 0.1, 0.2, 0.3, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert captura_no_topo(y, score, frac=0.2) == 1.0
    assert captura_no_topo(y, score[::-1], frac=0.2) == 0.0


def _base(abandono_t1, uf="RN", ano=2022):
    n = len(abandono_t1)
    return pd.DataFrame({
        "ano": ano, "SG_UF": uf, "abandono_med_t1": abandono_t1,
        "abandono_med": np.arange(n, dtype=float), "reprovacao_med": 1.0, "aprovacao_med": 90.0, "tdi_med": 20.0,
    })


def test_alvo_e_relativo_ao_estado_e_ano():
    # Mesmos valores em dois estados: o alvo marca o quartil superior dentro de cada um.
    df = pd.concat([_base([0, 2, 4, 6, 8, 10, 12, 14], "RN"), _base([0, 20, 40, 60, 80, 100, 120, 140], "CE")])
    out = enriquecer(df)
    assert out.groupby("SG_UF")["alvo"].sum().tolist() == [2, 2]


def test_alvo_ignora_abandono_zero():
    # Estado onde quase todas as escolas têm abandono 0: zero nunca vira alvo.
    out = enriquecer(_base([0, 0, 0, 0, 0, 0, 0, 5]))
    assert out.loc[out["abandono_med_t1"] == 0, "alvo"].sum() == 0
    assert out.loc[out["abandono_med_t1"] == 5, "alvo"].item() == 1
