"""
Quantum_Decisions_17.py

Biblioteka do przetwarzania danych z Banku Danych o Lasach (BDL), tworzenia operatorÃ³w (macierze obserwabli),
symulacji ewolucji czasowej i dopasowywania macierzy rotacji oraz Hamiltonianu.

Autor: Jan Kotlarz
Język: polski (dokumentacja w docstringach)

Struktura:
1) Funkcje wczytujące dane BDL i zapisujące CSV
2) Przygotowanie danych: wskażniki, dyskretyzacja
3) Budowa macierzy obserwabli, sprawdzenie komutacji
4) Symulacja: pomiar obserwabli, macierze przejść, obserwabli w stanach własnych
5) Analiza rok po roku: funkcje ewolucji, dopasowanie stacjonarnego Hamiltonianu, wskażnik dopasowania
6) PrzykÅ‚ad użycia (sekcja main wykorzystująca funkcje biblioteki)


"""

from typing import Tuple, List, Dict, Any
import numpy as np
import pandas as pd
from scipy.linalg import expm, eig
from scipy.optimize import minimize
import math
import matplotlib.pyplot as plt

# -----------------------
# 1. Funkcje wczytujące dane i zapisu
# -----------------------

def read_bdl_file(year: int, bdl_prefix: str = "bdl_data_", area_prefix: str = "bdl_data_area_", sep="\t") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Wczytuje pliki BDL dla danego roku.

    Parametry:
    - year: rok (np. 2019)
    - bdl_prefix: prefiks pliku z danymi BDL, domyślnie "bdl_data_"
    - area_prefix: prefiks pliku z powierzchniami, domyślnie "bdl_data_area_"
    - sep: separator pliku (domyślnie tab)

    Zwraca: (bdl_df, area_df)
    """
    bdl_path = f"{bdl_prefix}{year}.csv"
    area_path = f"{area_prefix}{year}.csv"
    bdl = pd.read_csv(bdl_path, sep=sep)
    area = pd.read_csv(area_path, sep=sep)

    # reset indexy i dodaj ID
    for df in (bdl, area):
        df.reset_index(drop=True, inplace=True)
        df["id"] = range(len(df))
    return bdl, area


def save_dataframe_to_csv(df: pd.DataFrame, out_path: str):
    """Zapisuje DataFrame do CSV (domyślnie przecinek)."""
    df.to_csv(out_path, index=False)


# -----------------------
# 2. Przygotowanie danych: wskażniki i dyskretyzacja
# -----------------------

def prepare_bdl(bdl: pd.DataFrame, area: pd.DataFrame) -> pd.DataFrame:
    """
    Przygotowuje tabelę BDL: mapuje powierzchnie, liczy species_area, volume i wiek.

    Zakłada istnienie kolumn: arodes_int_num, part_cd_act, volume, species_age
    Zwraca zmodyfikowany DataFrame bdl (kopię)
    """
    df = bdl.copy()
    areas_map = {row.arodes_int_num: row.sub_area for _, row in area.iterrows()}
    df["arode_area"] = df.arodes_int_num.map(areas_map).fillna(0.0)

    # kod ochronności (przykładowo: zaczyna się od 'OCHR')
    forest_funcs = {row.arodes_int_num: row.forest_func_cd for _, row in area.iterrows()}
    df["ochr"] = df.arodes_int_num.map(forest_funcs)
    df["ochr"] = df["ochr"].apply(lambda x: 1 if isinstance(x, str) and x.startswith("OCHR") else 0)

    df["part_cd_act"] = pd.to_numeric(df.get("part_cd_act", 0), errors='coerce').fillna(0)
    df["species_area"] = df["arode_area"] * df["part_cd_act"].astype(int) * 0.1
    df["species_volume"] = df["species_area"] * df.get("volume", 0)
    df["age"] = df.get("species_age", 0).fillna(0).astype(float)
    return df


def shannon_index_from_df(df: pd.DataFrame, species_col: str = "species_cd") -> Tuple[float, int, float]:
    """
    Oblicza indeks Shannona (H), liczbę gatunków i indeks normalizowany H/log(S).
    Zwraca (H, S, H_norm)
    """
    unique_species = df[species_col].unique()
    areas = np.array([df[df[species_col] == s].species_area.sum() for s in unique_species])
    asum = areas.sum()
    if asum == 0 or len(unique_species) <= 1:
        return 0.0, len(unique_species), 0.0
    p = areas / asum
    p = p[p > 0]
    H = -np.sum(p * np.log(p))
    H_norm = H / np.log(len(unique_species)) if len(unique_species) > 1 else 0.0
    return float(H), int(len(unique_species)), float(H_norm)


def co2_uptake_over_n_years(age: float, years: int = 100) -> float:
    """
    Prosty model pochłaniania CO2 w zależności od wieku (przykładowe pasma).  
    Zwraca sumę pochłonięcia (t/ha) w okresie `years` zaczynając od wieku `age`.
    """
    bands = [
        (0, 20, 6.0),
        (20, 60, 4.0),
        (60, 100, 2.0),
        (100, np.inf, 1.0)
    ]
    total = 0.0
    start = age
    end = age + years
    for b0, b1, rate in bands:
        lo = max(start, b0)
        hi = min(end, b1)
        if hi > lo:
            total += (hi - lo) * rate
    return float(total)


def compute_total_volume(df: pd.DataFrame) -> float:
    """Sumuje objętości gatunkÃ³w z kolumny 'volume' z wagą species_area."""
    return float((df.get("volume", 0) * df["species_area"]).sum())


def compute_thresholds(arr1: np.ndarray, arr2: np.ndarray, arr3: np.ndarray, t1: int = 25, t2: int = 75) -> Tuple[float,float,float,float,float,float]:
    """
    Oblicza progi percentylowe (t1,t2) dla trzech wektorów.
    Zwraca (a1_lo,a1_hi,a2_lo,a2_hi,a3_lo,a3_hi)
    """
    x = (
        np.percentile(arr1, t1), np.percentile(arr1, t2),
        np.percentile(arr2, t1), np.percentile(arr2, t2),
        np.percentile(arr3, t1), np.percentile(arr3, t2)
    )
    return x


def discretize_series(v: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Dyskretyzuje wektor do -1,0,1 według progÃ³w lo i hi."""
    return np.select([v <= lo, (v > lo) & (v <= hi), v > hi], [-1, 0, 1])


# -----------------------
# 3. Budowa macierzy obserwabli i sprawdzenie komutacji
# -----------------------

def q_diag_3() -> np.ndarray:
    """DiagonaÅ‚ Q = diag([-1,0,1]) (3x3)"""
    return np.diag([-1, 0, 1])


def expand_rotation_to_tensor(R3: np.ndarray) -> np.ndarray:
    """Rozszerza macierz 3x3 na tensor 9x9: I_3 × R3 (kron) zgodnie z poprzednim kodem."""
    I3 = np.eye(3)
    return np.kron(I3, R3)


def build_Q_tensors() -> Tuple[np.ndarray, np.ndarray]:
    """
    Zwraca stałe tensory Q1 i Q2 (9x9) z definicji:
    Q1_tensor = kron(Q_diag, I3)
    Q2_tensor = kron(I3, Q_diag)
    """
    Qd = q_diag_3()
    I3 = np.eye(3)
    Q1 = np.kron(Qd, I3)
    Q2 = np.kron(I3, Qd)
    return Q1, Q2


def build_transition_matrix_from_discretized(df: pd.DataFrame, from_col: str, to_col: str) -> np.ndarray:
    """
    Buduje 3x3 tabelę przejść (nieznormalizowaną) i zwraca znormalizowaną macierz przejść (wiersze sumują się do 1;
    tam gdzie suma 0 -> zostaje 0). Indeksacja w porządku [-1,0,1].
    """
    idx = [-1, 0, 1]
    t = pd.DataFrame(0, index=idx, columns=idx)
    for s0 in idx:
        for s1 in idx:
            t.loc[s0, s1] = ((df[from_col] == s0) & (df[to_col] == s1)).sum()
    norm = t.div(t.sum(axis=1).replace(0, 1), axis=0)
    return norm.values.astype(float)


def observables_commute(A: np.ndarray, B: np.ndarray, atol: float = 1e-8) -> bool:
    """Sprawdza czy [A,B] = 0 (macierz komutatora bliska zeru)"""
    C = A @ B - B @ A
    return np.allclose(C, np.zeros_like(C), atol=atol)


# -----------------------
# 4. Symulacja: pomiary, macierze przejÅ›Ä, obserwabli w stanach wÃasnych
# -----------------------

def measure_observable(state_index: int, observable: np.ndarray) -> float:
    """
    Pomiar prostej obserwabli (zakładamy bazę kanoniczną): zwraca wartość eigenvalue
    dla indeksu stanu (-1->0, 0->1, 1->2 w indeksie macierzy 3x3).
    """
    map_idx = { -1:0, 0:1, 1:2 }
    i = map_idx[state_index]
    # wartoÅ›ci diagonalne obserwabli Q_diag_3
    return observable[i, i]


def observable_in_eigenbasis(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Przekształca B do bazy własnej A: U^∗ B U
    gdzie kolumny U to wektory własne macierzy A.
    Zwraca macierz B w bazie wlasnej A.
    """
    vals, vecs = eig(A)
    U = vecs
    return U.conj().T @ B @ U


def compute_transition_tensor(T3: np.ndarray) -> np.ndarray:
    """
    Rozszerza 3x3 macierz przejsć do tensora 9x9 przez kron(I3,T3)
    """
    return np.kron(np.eye(3), T3)


# -----------------------
# 5. Analiza rok po roku, ewolucja czasowa i dopasowanie Hamiltonianu
# -----------------------

def evolve_density_by_H(H: np.ndarray, V0: np.ndarray, years: List[int], year0: int, time_unit_seconds: float = 1.0, hbar: float = 1.0) -> Dict[int, np.ndarray]:
    """
    Ewolucja macierzy gęstości/operatora V0 pod stacjonarnym H:
    V(t) = U(t) V0 U(t)^†, U = exp(-i/hbar H dt)

    Parametry:
      H: hermitowska macierz (n x n)
      V0: macierz początkowa (n x n)
      years: lista lat do obliczenia (np. [2015,2016,...])
      year0: rok początkowy odpowiadający V0
      time_unit_seconds: ile sekundy odpowiada 1 "roku" (domyślnie 1 -> jednostka = rok)
      hbar: stała Plancka (domyślnie 1 dla stabilności numerycznej)

    Zwraca: słownik {rok: V(t)}
    """
    out = {}
    for y in years:
        dt = (y - year0) * time_unit_seconds
        U = expm((-1j / hbar) * H * dt)
        Vt = U @ V0 @ U.conj().T
        out[y] = Vt
    return out


def fit_hamiltonian_to_rotations(V_by_year: Dict[int, np.ndarray], year0: int, time_unit_seconds: float = 1.0, hbar: float = 1.0, maxiter:int=500) -> Tuple[np.ndarray, Any]:
    """
    Dopasowuje hermitowski Hamiltonian H (3x3) tak, aby ewolucja V0 -> V(year) pasowała do danych.
    Parametry podobne do skryptu hamiltonian.py z przesÅanych plikÃ³w.

    Zwraca (H_fit, opt_result)
    """
    years = sorted(V_by_year.keys())
    V0 = V_by_year[year0]

    def vector_to_H(p):
        if len(p) != 9:
            raise ValueError("Oczekiwane 9 parametrów dla 3x3 macierzy Hermitowskiej.")
        d0, d1, d2 = p[0], p[1], p[2]
        re01, im01 = p[3], p[4]
        re02, im02 = p[5], p[6]
        re12, im12 = p[7], p[8]

        H = np.zeros((3,3), dtype=complex)
        H[0,0] = d0
        H[1,1] = d1
        H[2,2] = d2

        H[0,1] = re01 + 1j*im01
        H[1,0] = re01 - 1j*im01

        H[0,2] = re02 + 1j*im02
        H[2,0] = re02 - 1j*im02

        H[1,2] = re12 + 1j*im12
        H[2,1] = re12 - 1j*im12

        return H

    def loss(p):
        H = vector_to_H(p)
        total = 0.0
        for y in years:
            dt = (y - year0) * time_unit_seconds
            U = expm((-1j / hbar) * H * dt)
            V_pred = U @ V0 @ U.conj().T
            diff = V_pred - V_by_year[y]
            total += np.linalg.norm(diff, 'fro')**2
        return total

    p0 = 1e-3 * np.random.randn(9)
    res = minimize(loss, p0, method='BFGS', options={'maxiter':maxiter, 'disp': False})
    H_fit = vector_to_H(res.x)
    return H_fit, res


def rotation_angles_from_R3(R3: np.ndarray) -> Tuple[float,float,float]:
    """
    Ekstrahuje kąty (phi, theta, rho) z dekompozycji Z*Y*X przy zaÅ‚ożeniu konwencji użytej w kodzie 'rotation_matrix_from_angles'.
    Funkcja zwraca kąty w radianach.

    Uwaga: odwrotne mapowanie nie jest jednoznaczne dla wszystkich macierzy rotacji (konwencje Eulerowe), ale ten solver stosuje prostą metodę arctan/asin.
    """
    # Zakładamy: R = Rz(phi) Ry(theta) Rx(rho)
    # Z R[0,2] = cos(phi)*sin(theta)
    theta = math.asin(max(-1.0, min(1.0, R3[0,2])))
    if abs(math.cos(theta)) < 1e-8:
        # singuralnosc: theta ~ +-pi/2
        phi = math.atan2(-R3[1,0], R3[1,1])
        rho = 0.0
    else:
        phi = math.atan2(R3[1,0], R3[0,0])
        rho = math.atan2(R3[2,1], R3[2,2])
    return phi, theta, rho


# -----------------------
# 6. Funkcje wizualizacyjne i pomocnicze
# -----------------------

def plot_matrix(M: np.ndarray, title: str = "Macierz", annotate: bool = True, cmap: str = 'viridis'):
    """Rysuje macierz M z kolorami i (opcjonalnie) podpisami elementÃ³w."""
    plt.imshow(np.real(M), aspect='equal')
    plt.colorbar()
    plt.title(title)
    if annotate:
        n, m = M.shape
        for i in range(n):
            for j in range(m):
                plt.text(j, i, f"{np.real(M[i,j]):.3f}", ha='center', va='center', color='white', fontsize=8)
    plt.show()


def plot_angles_over_time(df_angles: pd.DataFrame, cols: List[str] = ["phi_deg","theta_deg","rho_deg"]):
    """Rysuje ewolucjÄ kątÃ³w w czasie na podstawie DataFrame z kolumnami year i phi_deg, theta_deg, rho_deg."""
    fig, ax = plt.subplots(figsize=(8,4))
    for c in cols:
        if c in df_angles.columns:
            ax.plot(df_angles.index, df_angles[c], marker='o', label=c)
    ax.set_xlabel('Rok')
    ax.set_ylabel('Kąt (stopnie)')
    ax.legend()
    ax.grid(True)
    plt.show()


# -----------------------
# Przykład uzycia: 
# - wczytaj wyniki dopasowania rotacji (rotation_fit_results_by_year.csv)
# - wczytaj dopasowany Hamiltonian (jeśli jest)
# - narysuj macierze i kąty
# -----------------------
if __name__ == '__main__':
    print("PrzykÅ‚ad: budowanie macierzy i dopasowanie Hamiltonianu — przykÅ‚adowe ")
    try:
        rot_df = pd.read_csv('rotation_fit_results_by_year.csv', index_col='year')
        print("Wczytano rotation_fit_results_by_year.csv")
        print(rot_df.head())
        plot_angles_over_time(rot_df)
    except Exception as e:
        print("Brak pliku rotation_fit_results_by_year.csv w katalogu. Aby uruchomiÄ‡ peÅ‚ny przyklad, wygeneruj wczeÅ›niejsze wyniki (np. uruchom skrypt quantum5.py).\n", e)

    # Jeśli użytkownik ma plik z macierzami V po latach, moÅ¼na dopasowaÄ‡ H:
    try:
        Vdf = pd.read_csv('rotation_fit_results_by_year.csv')
        # przykÅadowo - zakÅadamy, Å¼e używamy trzech kolumn tworzyÅych macierz 3x3 dla kaÅ¼dego roku
        # ale konkretny format danych zaleÅ¼y od użytkownika
        print('Gotowe funkcje do dopasowania H i rysowania dostarczone w module.')
    except Exception:
        pass


# KONIEC PLIKU
