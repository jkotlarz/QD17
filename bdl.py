import os
import pandas as pd
import numpy as np
import re

# ------------------------------------------------------------
# Prepare source csv files
# ------------------------------------------------------------


def generate_bdl_csv_files(folder_path: str):
    """
    Funkcja generuje dwa pliki CSV:
      - bdl_data_<year>.csv
      - bdl_data_area_<year>.csv
    na podstawie plików:
      - f_storey_species.txt
      - f_subarea.txt
    
    Parametry:
        folder_path (str): pełna ścieżka do folderu źródłowego
                           (np. r".../BDL_12_24_TORUN_2025")
    
    Zwraca:
        (str, str) → ścieżki do wygenerowanych plików CSV
    """
    
    # --- 1. Detekcja roku z nazwy folderu ---
    match = re.search(r"(\d{4})", os.path.basename(folder_path))
    if not match:
        raise ValueError("Nie znaleziono roku (YYYY) w nazwie folderu.")
    year = match.group(1)

    # --- 2. Ścieżki wejściowe ---
    path_storey = os.path.join(folder_path, "f_storey_species.txt")
    path_subarea = os.path.join(folder_path, "f_subarea.txt")

    if not os.path.isfile(path_storey):
        raise FileNotFoundError(f"Brak pliku: {path_storey}")

    if not os.path.isfile(path_subarea):
        raise FileNotFoundError(f"Brak pliku: {path_subarea}")

    # --- 3. Przetwarzanie f_storey_species.txt ---
    df = pd.read_csv(path_storey, sep="\t", dtype=str)

    num_cols = ['species_age', 'height', 'bhd', 'volume']
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')

    df = df[
        (df.storey_cd.str.startswith("DRZEW")) &
        (df.volume > 0.0)
    ]

    out_bdl = os.path.join(folder_path, f"bdl_data.csv")
    df.to_csv(out_bdl, sep="\t", index=False)

    # --- 4. Przetwarzanie f_subarea.txt ---
    df_area = pd.read_csv(path_subarea, sep="\t", dtype=str)
    df_area = df_area[df_area.area_type_cd.str.startswith("D-")]

    out_area = os.path.join(folder_path, f"bdl_data_area.csv")
    df_area.to_csv(out_area, sep="\t", index=False)

    # --- RETURN ---
    return out_bdl, out_area



# ------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------
def co2_uptake_over_n_years(age, years=100):
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
    return total


def co2_for_dataframe_sample(df_sample, years=100):
    co2_per_ha = df_sample["age"].apply(lambda a: co2_uptake_over_n_years(a, years=years))
    total_co2 = (co2_per_ha * df_sample["species_area"]).sum()
    return total_co2


def shannon_index_from_df(df):
    unique_species = df.species_cd.unique()
    areas = np.array([df[df.species_cd == s].species_area.sum() for s in unique_species])
    asum = areas.sum()
    if asum == 0 or len(unique_species) <= 1:
        return [0.0, len(unique_species), 0.0]
    p = areas / asum
    p = p[p > 0]
    H = -np.sum(p * np.log(p))
    return [H, len(unique_species), H / np.log(len(unique_species))]


# ------------------------------------------------------------
# Main strategy generation function
# ------------------------------------------------------------
def generate_strategies(
    rec_in_sample: int,
    folder: str,
    target_volume: float,
    target_shannon: float,
    target_protection: float,
    target_co2: float,
    n_strategies: int,
    min_age: int = 80
):
    """
    Parameters
    ----------
    folder : str
        Folder containing CSV files.
    target_volume : float
    target_shannon : float
    target_protection : float
    target_co2 : float
    n_strategies : int
    min_age : int
        Minimum species_age for filtering.

    Returns
    -------
    df_strategies : DataFrame
        strategy_id, Q1v, Q2v, Q3v, Q4v

    df_records : DataFrame
        All records belonging to each strategy.
    """

    # ---------------- Load base files -----------------
    bdl_path = os.path.join(folder, "bdl_data.csv")
    area_path = os.path.join(folder, "bdl_data_area.csv")

    bdl = pd.read_csv(bdl_path, sep="\t")
    area = pd.read_csv(area_path, sep="\t")

    bdl = bdl.reset_index(drop=True)
    bdl["id"] = range(len(bdl))
    area = area.reset_index(drop=True)

    # ---------------- Mapping area & functions ----------------
    areas_map = {row.arodes_int_num: row.sub_area for _, row in area.iterrows()}
    bdl["arode_area"] = bdl.arodes_int_num.map(areas_map).fillna(0.0)

    forest_funcs = {row.arodes_int_num: row.forest_func_cd for _, row in area.iterrows()}
    bdl["ochr"] = bdl.arodes_int_num.map(forest_funcs)
    bdl["ochr"] = bdl["ochr"].apply(lambda x: 1 if isinstance(x, str) and x.startswith("OCHR") else 0)

    bdl["part_cd_act"] = pd.to_numeric(bdl.get("part_cd_act", 0), errors="coerce").fillna(0.0)
    bdl["species_area"] = bdl["arode_area"] * bdl["part_cd_act"].astype(int) * 0.1
    bdl["species_volume"] = bdl["species_area"] * bdl.get("volume", 0)
    bdl["age"] = bdl.get("species_age", 0).fillna(0).astype(float)

    # ---------------- Filtering ----------------
    filtered_bdl = bdl[bdl['species_age'] >= min_age]

    samples = []

    # ---------------- Generate strategies ----------------
    for _ in range(n_strategies):
        sample = filtered_bdl.sample(rec_in_sample)
        samples.append(sample)

    rests = [bdl.loc[~bdl['id'].isin(s['id'])] for s in samples]

    # ---------------- Compute indicators ----------------
    s_vol = [-np.abs(s['species_volume'].sum() - target_volume) for s in samples]
    s_div = [shannon_index_from_df(r)[2] - target_shannon for r in rests]
    s_och = [r.ochr.sum() - target_protection for r in rests]
    s_co2 = [co2_for_dataframe_sample(r) - target_co2 for r in rests]

    df_strategies = pd.DataFrame({
        "strategy_id": range(n_strategies),
        "Q1v": s_vol,
        "Q2v": s_div,
        "Q3v": s_och,
        "Q4v": s_co2
    })

    return df_strategies, samples


def transform_Q_columns(df: pd.DataFrame, rules: dict = None):
    """
    Transform numeric Q* columns in df.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    rules : dict, optional
        Dictionary of rules by column:
        {
            "Q1": [[cut_values], [target_values]],
            "Q2": [[cut_values], [target_values]],
            ...
        }
        - values in cut_values define the bin edges (without -inf, inf)
        - target_values define the mapping for bins in order
    
    Returns
    -------
    df_transformed : pd.DataFrame
        DataFrame with transformed Q* columns.
    """

    df = df.copy()
    q_columns = [col for col in df.columns if col.startswith("Q")]

    for col in q_columns:

        if rules and col in rules:
            # User-defined rules
            cuts, targets = rules[col]
            
            # Create full bin edges with -inf / +inf
            bins = [-np.inf] + list(cuts) + [np.inf]

            # Use pandas cut
            df[col] = pd.cut(df[col], bins=bins, labels=targets).astype(float)

        else:
            # Default: median split → [-1, +1]
            med = df[col].median()
            df[col] = np.where(df[col] < med, -1, 1)

    return df
