# -*- coding: utf-8 -*-
"""
Quantum Harvesting Analysis — modularny refactor
Autor: Jan Kotlarz
Data: 2025-11-10

Opis:
- Sekcja 1: Funkcje do przekształcania plików tekstowych BDL do DataFrame
- Sekcja 2: Dyskretyzacja wskaźników (Q1v..Q4v → Q1..Q4)
- Sekcja 3: Modelowanie kwantowe (macierze przejść, dopasowanie rotacji, błędy, oczekiwania)
- Sekcja 4: Wizualizacja wyników (macierze, błędy, kąty, wartości oczekiwane)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ============================================================
# SEKCJA 1 — WCZYTANIE I PRZEKSZTAŁCANIE DANYCH Z BDL
# ============================================================

def load_bdl_data(path_data, path_area):
    """Wczytuje pliki tekstowe BDL i łączy dane obszarowe."""
    bdl = pd.read_csv(path_data, sep="\t")
    area = pd.read_csv(path_area, sep="\t")

    for df in (bdl, area):
        df.reset_index(drop=True, inplace=True)
        df["id"] = range(len(df))

    # Mapowanie powierzchni i funkcji lasu
    areas_map = {row.arodes_int_num: row.sub_area for _, row in area.iterrows()}
    forest_funcs = {row.arodes_int_num: row.forest_func_cd for _, row in area.iterrows()}

    bdl["arode_area"] = bdl.arodes_int_num.map(areas_map).fillna(0.0)
    bdl["ochr"] = bdl.arodes_int_num.map(forest_funcs)
    bdl["ochr"] = bdl["ochr"].apply(lambda x: 1 if isinstance(x, str) and x.startswith("OCHR") else 0)

    bdl["part_cd_act"] = pd.to_numeric(bdl["part_cd_act"], errors='coerce').fillna(0.0)
    bdl["species_area"] = bdl["arode_area"] * bdl["part_cd_act"].astype(int) * 0.1
    bdl["species_volume"] = bdl["species_area"] * bdl["volume"]
    bdl["age"] = bdl["species_age"].fillna(0).astype(float)
    return bdl


def co2_uptake_over_n_years(age, years=100):
    """Model pochłaniania CO2 (t/ha) przez las przez N lat."""
    bands = [(0, 20, 6.0), (20, 60, 4.0), (60, 100, 2.0), (100, np.inf, 1.0)]
    total = 0.0
    start, end = age, age + years
    for b0, b1, rate in bands:
        lo, hi = max(start, b0), min(end, b1)
        if hi > lo:
            total += (hi - lo) * rate
    return total


def co2_for_dataframe_sample(df, years=100):
    """Łączna ilość pochłoniętego CO2 dla próbki df."""
    co2_per_ha = df["age"].apply(lambda a: co2_uptake_over_n_years(a, years=years))
    return (co2_per_ha * df["species_area"]).sum()


def shannon_index_from_df(df):
    """Zwraca [H, liczba_gatunków, znormalizowany_H]."""
    species = df.species_cd.unique()
    areas = np.array([df[df.species_cd == s].species_area.sum() for s in species])
    asum = areas.sum()
    if asum == 0 or len(species) <= 1:
        return [0.0, len(species), 0.0]
    p = areas / asum
    H = -np.sum(p * np.log(p))
    return [H, len(species), H / np.log(len(species))]


def total_volume(df):
    """Całkowita objętość drzewostanu (m3)."""
    return (df.volume * df.species_area).sum()


# ============================================================
# SEKCJA 2 — DYKSRETYZACJA WSKAŹNIKÓW
# ============================================================

def compute_thresholds(arr, t1=15, t2=50):
    """Zwraca dolny i górny próg (percentyle)."""
    return np.percentile(arr, [t1, t2])


def discretize_values(df, varname, t1=15, t2=50):
    """Dyskretyzacja do wartości {-1, 0, 1}."""
    lo, hi = compute_thresholds(df[varname].values, t1, t2)
    return np.select(
        [df[varname] <= lo, (df[varname] > lo) & (df[varname] <= hi), df[varname] > hi],
        [-1, 0, 1]
    )


# ============================================================
# SEKCJA 3 — MODELOWANIE KWANTOWE
# ============================================================

A_diag = np.diag([-1, 0, 1])

def rotation_matrix_from_angles(phi, theta, rho):
    """Macierz rotacji 3D (Euler Z-Y-X)."""
    Rz = np.array([
        [np.cos(phi), -np.sin(phi), 0],
        [np.sin(phi),  np.cos(phi), 0],
        [0, 0, 1]
    ])
    Ry = np.array([
        [ np.cos(theta), 0, np.sin(theta)],
        [ 0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rho), -np.sin(rho)],
        [0, np.sin(rho),  np.cos(rho)]
    ])
    return Rz @ Ry @ Rx


def transition_matrix(df, var, ref_var, t1=15, t2=50):
    """Tworzy macierz przejść (3x3) między zmiennymi."""
    vals = df[ref_var + "v"].values
    lo, hi = compute_thresholds(vals, t1, t2)
    df["new_state"] = np.select([vals <= lo, (vals > lo) & (vals <= hi), vals > hi], [-1, 0, 1])

    transitions = pd.DataFrame(0, index=[-1, 0, 1], columns=[-1, 0, 1], dtype=float)
    for s0 in [-1, 0, 1]:
        for s1 in [-1, 0, 1]:
            transitions.loc[s0, s1] = ((df[var] == s0) & (df["new_state"] == s1)).sum()
    tm = transitions.div(transitions.sum(axis=1).replace(0, 1), axis=0) * 100.0
    return tm.round(4)


def fit_rotation(A_input):
    """Dopasowuje macierz rotacji do A_input."""
    A0 = A_diag.copy()

    def objective(angles):
        R = rotation_matrix_from_angles(*angles)
        return np.linalg.norm(A_input - R @ A0 @ R.T)

    res = minimize(objective, x0=[0.0, 0.0, 0.0], bounds=[(-np.pi, np.pi)] * 3)
    phi, theta, rho = res.x
    R_best = rotation_matrix_from_angles(phi, theta, rho)
    A_best = R_best @ A0 @ R_best.T
    return {
        "A_best": A_best,
        "R_best": R_best,
        "angles_rad": [phi, theta, rho],
        "angles_deg": np.degrees([phi, theta, rho]),
        "fit_error": np.linalg.norm(A_input - A_best)
    }


def expected_value(A, psi):
    psi = np.array(psi, dtype=complex)
    return np.real(np.vdot(psi, A @ psi))


def uncertainty(A, psi):
    exp_A = expected_value(A, psi)
    exp_A2 = expected_value(A @ A, psi)
    val = exp_A2 - exp_A**2
    return np.sqrt(val) if val >= 0 else 0.0


# ============================================================
# SEKCJA 4 — WIZUALIZACJA
# ============================================================

def plot_matrix(ax, M, title, cmap='cividis', fmt='.3f'):
    im = ax.imshow(M, cmap=cmap, interpolation='nearest')
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels([r"$|{-1}\rangle$", r"$|0\rangle$", r"$|1\rangle$"])
    ax.set_yticklabels([r"$|{-1}\rangle$", r"$|0\rangle$", r"$|1\rangle$"])
    ax.set_title(title)
    for (i, j), val in np.ndenumerate(M):
        ax.text(j, i, f"{val:{fmt}}", ha='center', va='center', color='white',
                fontsize=10, fontweight='bold')
    plt.colorbar(im, ax=ax, fraction=0.046)


# ============================================================
# PRZYKŁAD UŻYCIA
# ============================================================

if __name__ == "__main__":
    # 1. Wczytaj dane
    bdl = load_bdl_data("bdl_data.csv", "bdl_data_area.csv")

    # 2. Utwórz wskaźniki
    df = pd.DataFrame({
        "Q1v": [total_volume(bdl)],
        "Q2v": [shannon_index_from_df(bdl)[2]],
        "Q3v": [bdl.ochr.sum()],
        "Q4v": [co2_for_dataframe_sample(bdl)]
    })

    # 3. Dyskretyzacja
    for q in ["Q1v", "Q2v", "Q3v", "Q4v"]:
        df[q.replace("v", "")] = discretize_values(df, q)

    # 4. Macierz przejść i rotacja
    tm = transition_matrix(df, "Q1", "Q2")
    A_input = tm.values / 100.0 @ A_diag @ (tm.values / 100.0).T
    fit = fit_rotation(A_input)

    # 5. Wykres macierzy
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    plot_matrix(ax, tm.values, "Macierz przejść Q1→Q2")
    plt.show()

    print("Kąty dopasowania (°):", fit["angles_deg"])
    print("Błąd dopasowania:", fit["fit_error"])
