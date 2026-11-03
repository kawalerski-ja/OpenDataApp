from __future__ import annotations

from pathlib import Path
import re
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from shapely.geometry import Point

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "krakow_master_grid_komplet.parquet"
PRICE_FILE = ROOT / "data" / "rejestr_cen.csv"
METRIC_CRS = "EPSG:2178"
RADIUS_M = 500
GROUPS = [
    "Transport", "Edukacja", "Zdrowie", "Sport", "Usługi publiczne",
    "Środowisko", "Ceny", "Dostępność",
]
DEFAULT_WEIGHTS = {
    "Transport": 5, "Edukacja": 3, "Zdrowie": 3, "Sport": 3,
    "Usługi publiczne": 2, "Środowisko": 3,
    "Ceny": 0, "Dostępność": 0,
}

# --- NOWE: strefy miasta (odległość od Rynku Głównego) -----------------------
CENTER_LONLAT = (19.93732, 50.06168)          # Rynek Główny
RING_NAMES = ["Śródmieście", "Miasto", "Obrzeża"]
RING_EDGES_KM = (2.5, 6.5)                    # granice stref: <2,5 km | <6,5 km | dalej
BEARING_NAMES = ["północ", "północny wschód", "wschód", "południowy wschód",
                 "południe", "południowy zachód", "zachód", "północny zachód"]
# Wyostrza wpływ suwaków: waga 5 vs 1 ma realnie większe znaczenie (waga**SHARPNESS).
SHARPNESS = 1.5

FLAG_FEATURES = {
    "stops": "czy_przystanek_do_100m",
    "cycle": "czy_zawiera_ciag_rowerowy",
    "education": "czy_zawiera_edukacje",
    "health": "czy_zawiera_zdrowie",
    "sport": "czy_zawiera_sport",
    "institution": "czy_zawiera_instytucje",
    "water": "czy_zawiera_pitnik",
    "toilets": "czy_zawiera_toalete",
    "police": "czy_zawiera_policje",
    "mops": "czy_zawiera_mops",
    "cas": "czy_zawiera_cas",
    "green": "czy_zielony_korytarz",
}
# Jawne wagi WEWNĄTRZ wymiaru; nie premiujemy wielokrotnie jednej usługi.
INTERNAL_WEIGHTS = {
    "Transport": {"stops": 0.60, "cycle": 0.25, "cycle_amount": 0.15},
    "Edukacja": {"education": 1},
    "Zdrowie": {"health": 1},
    "Sport": {"sport": 1},
    "Usługi publiczne": {
        "institution": .30, "mops": .15, "cas": .15,
        "police": .10, "water": .15, "toilets": .15,
    },
    "Środowisko": {"green": 1},
    "Ceny": {"price_utility": 1},
    "Dostępność": {"access_utility": 1},
}
# Pola opisowe: nazwy i kategorie NIE są osobnymi punktami rankingowymi.
DESCRIPTIVE_FIELDS = [
    "typy_infrastruktury_rowerowej", "przystanki_do_100m",
    "kategorie_edukacji", "nazwy_edukacja", "nazwy_zdrowie",
    "nazwy_sport", "rodzaje_sportu", "dyscypliny_sportowe",
    "nazwy_instytucje", "nazwy_mops", "nazwy_cas",
    "nazwy_korytarzy_zieleni", "mpzp_przeznaczenia", "mpzp_rok_planu",
    "max_halas_dB", "max_halas_szyny_dB",
    "wentyl_predkosc_v4m", "wentyl_potencjal",
    "score_niepelnosprawni", "cena_sr_za_m2",
    "cena_sr_calkowita", "cena_sr_powierzchnia", "cena_data_aktualnosci",
]


def _weights(weights: dict) -> np.ndarray:
    w = np.array([float(weights.get(g, 0)) for g in GROUPS], float)
    if not np.isfinite(w).all() or (w < 0).any() or w.sum() == 0:
        raise ValueError("Wagi muszą być nieujemne, skończone; przynajmniej jedna > 0")
    return w / w.sum()


def _weighted(matrix, weights):
    mat = np.asarray(matrix, float)
    valid = np.isfinite(mat)
    available = valid @ weights
    numerator = np.where(valid, mat, 0.0) @ weights
    scores = np.divide(numerator, available, out=np.full(len(mat), np.nan), where=available > 0)
    return 100 * scores, available


def _binary(df, name):
    if name not in df:
        return np.full(len(df), np.nan)
    series = pd.to_numeric(df[name], errors="coerce").to_numpy(float)
    invalid = np.isfinite(series) & ~np.isin(series, [0, 1])
    if invalid.any():
        warnings.warn(f"Nieprawidłowe flagi w {name}: {invalid.sum()} — traktuję jako brak")
        series[invalid] = np.nan
    return series


def _adjacency(coords, radius):
    neighbors = cKDTree(coords).query_ball_point(coords, radius)
    lengths = np.fromiter(map(len, neighbors), dtype=np.int32, count=len(coords))
    rows = np.repeat(np.arange(len(coords)), lengths)
    cols = np.concatenate([np.asarray(k, dtype=np.int32) for k in neighbors])
    return csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                      shape=(len(coords), len(coords)))


def _local_mean(values, adjacency):
    v = np.asarray(values, float)
    good = np.isfinite(v)
    denom = np.asarray(adjacency @ good.astype(float)).ravel()
    num = np.asarray(adjacency @ np.where(good, v, 0.0)).ravel()
    return np.divide(num, denom, out=np.full(len(v), np.nan), where=denom > 0)


def _nearest_utility(coords, flag, near_m=100, far_m=1200):
    """Użyteczność odległości do centroidu heksagonu z obiektem; nie czas dojścia."""
    good = np.asarray(flag, float)
    yes = np.flatnonzero(good == 1)
    if len(yes) == 0:
        return np.full(len(coords), np.nan) if np.isnan(good).any() else np.zeros(len(coords))
    distances = cKDTree(coords[yes]).query(coords)[0]
    return np.clip((far_m - np.maximum(distances, near_m)) / (far_m - near_m), 0, 1)


def _combine(parts, internal):
    data = np.column_stack([parts[key] for key in internal])
    weights = np.array(list(internal.values()), float)
    weights /= weights.sum()
    return _weighted(data, weights)[0] / 100


def _robust_utility(arr, higher_is_better=True):
    """Odwzorowanie 5.–95. percentyla na 0–1; brak danych pozostaje NaN."""
    values = np.asarray(arr, float)
    good = values[np.isfinite(values)]
    if len(good) < 20:
        return np.full(len(values), np.nan)
    lo, hi = np.quantile(good, [.05, .95])
    if hi <= lo:
        return np.full(len(values), np.nan)
    utility = np.clip((values - lo) / (hi - lo), 0, 1)
    return utility if higher_is_better else 1 - utility


def _percentile(values, groups=None):
    """Rangi percentylowe 0–1 (NaN zostaje NaN). Rozbija remisy i 'sufit' wyników,
    przez które wygrywało w kółko to samo centrum.
    groups: opcjonalnie etykiety stref -> percentyl liczony tylko wśród podobnych okolic."""
    s = pd.Series(np.asarray(values, float))
    if groups is None:
        return s.rank(pct=True, method="average").to_numpy()
    return s.groupby(np.asarray(groups), sort=False).rank(pct=True, method="average").to_numpy()


def _district_number(value):
    if pd.isna(value):
        return np.nan
    s = str(value).strip().upper()
    roman = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6,
             "VII": 7, "VIII": 8, "IX": 9, "X": 10, "XI": 11, "XII": 12,
             "XIII": 13, "XIV": 14, "XV": 15, "XVI": 16, "XVII": 17, "XVIII": 18}
    head = re.split(r"[,;\s\-]", s)[0]
    if head in roman:
        return roman[head]
    m = re.search(r"\d+", s)
    if m and 1 <= int(m.group()) <= 18:
        return int(m.group())
    return np.nan


def prepare_district_prices(path=PRICE_FILE, years=(2024, 2025, 2026),
                            min_local_count=10):
    """Ceny rynku wtórnego lokali mieszkalnych, ważone liczbą lokali.

    Nie oznaczają ceny konkretnej nieruchomości. W przypadku powtarzających
    się/zbiorczych statystyk źródłowych agregacja wymaga weryfikacji definicji.
    """
    p = Path(path)
    if not p.is_file():
        return pd.DataFrame(columns=["district_no", "price_m2", "n_locals", "n_records", "last_year"])
    raw = pd.read_csv(p, sep=";", encoding="utf-8-sig", low_memory=False)
    required = ["data_zaw_year", "sr_cena_m2", "lkl_count", "ozn_jedn_u"]
    missing = [c for c in required if c not in raw]
    if missing:
        raise ValueError(f"W rejestrze cen brakuje kolumn {missing}")
    raw["year"] = pd.to_numeric(raw["data_zaw_year"], errors="coerce")
    raw["price"] = pd.to_numeric(raw["sr_cena_m2"], errors="coerce")
    raw["locals"] = pd.to_numeric(raw["lkl_count"], errors="coerce")
    raw["district_no"] = raw["ozn_jedn_u"].map(_district_number)
    mask = raw["year"].isin(years) & raw["price"].gt(0) & raw["locals"].gt(0)
    if "rodzaj_rynku" in raw:
        mask &= raw["rodzaj_rynku"].astype(str).str.contains("wtórny", case=False, na=False)
    if "lkl_funkcja" in raw:
        mask &= raw["lkl_funkcja"].astype(str).str.contains("mieszkalne", case=False, na=False)
    raw = raw.loc[mask & raw["district_no"].notna()].copy()
    if raw.empty:
        return pd.DataFrame(columns=["district_no", "price_m2", "n_locals", "n_records", "last_year"])
    raw["weighted_price"] = raw["price"] * raw["locals"]
    summary = raw.groupby("district_no", as_index=False).agg(
        weighted_total=("weighted_price", "sum"), n_locals=("locals", "sum"),
        n_records=("price", "size"), last_year=("year", "max"))
    summary["price_m2"] = summary["weighted_total"] / summary["n_locals"]
    summary.loc[summary["n_locals"] < min_local_count, "price_m2"] = np.nan
    return summary[["district_no", "price_m2", "n_locals", "n_records", "last_year"]]


def _districts_for_centers(metric_centers, district_path, district_column):
    if district_path is None:
        return np.full(len(metric_centers), np.nan)
    boundary = gpd.read_file(district_path)
    if boundary.crs is None or district_column not in boundary:
        raise ValueError("Warstwa dzielnic musi mieć CRS i kolumnę numerów/nazw dzielnic")
    boundary = boundary.to_crs(METRIC_CRS).copy()
    boundary["district_no"] = boundary[district_column].map(_district_number)
    boundary = boundary.dropna(subset=["district_no", "geometry"])
    if boundary.empty:
        raise ValueError("Nie można odczytać numerów dzielnic z warstwy granic")
    pts = gpd.GeoDataFrame({"pos": np.arange(len(metric_centers))},
                           geometry=metric_centers, crs=METRIC_CRS)
    joined = gpd.sjoin(pts, boundary[["district_no", "geometry"]], how="left", predicate="intersects")
    joined = joined.sort_values("pos").drop_duplicates("pos")
    return joined.set_index("pos")["district_no"].reindex(range(len(pts))).to_numpy(float)


def prepare_city(path=DATA_FILE, radius_m=RADIUS_M,
                 district_path=None, district_column="district_no",
                 price_path=PRICE_FILE,
                 ventilation_higher_is_better=None,
                 accessibility_higher_is_better=None,
                 accessibility_scale=None):
    """Przygotowanie cech: wywołać raz, wyniki cache'ować w Streamlit."""
    if radius_m <= 0:
        raise ValueError("Promień musi być dodatni")
    gdf = gpd.read_parquet(path)
    if gdf.empty or gdf.crs is None or "GRID_ID" not in gdf:
        raise ValueError("Nieprawidłowy Parquet: brak CRS, GRID_ID lub wierszy")
    if gdf["GRID_ID"].isna().any() or gdf["GRID_ID"].duplicated().any():
        raise ValueError("GRID_ID musi być niepusty i unikatowy")
    if gdf.geometry.isna().any() or gdf.geometry.is_empty.any():
        raise ValueError("Brakujące lub puste geometrie")
    geom = gdf.to_crs(METRIC_CRS).geometry.centroid
    coords = np.column_stack((geom.x.to_numpy(), geom.y.to_numpy()))
    if not np.isfinite(coords).all():
        raise ValueError("Nieprawidłowe współrzędne")
    ll = gpd.GeoSeries(geom, crs=METRIC_CRS).to_crs("EPSG:4326")
    adj = _adjacency(coords, radius_m)
    result = pd.DataFrame({
        "GRID_ID": gdf["GRID_ID"].astype(str).to_numpy(),
        "x": coords[:, 0], "y": coords[:, 1],
        "lat": ll.y.to_numpy(), "lon": ll.x.to_numpy(),
        "neighbors_n": np.asarray(adj.sum(axis=1)).ravel().astype(int),
    })

    # NOWE: strefa miasta, odległość i kierunek od Rynku (do nazw i filtrów w interfejsie)
    center = gpd.GeoSeries([Point(*CENTER_LONLAT)], crs="EPSG:4326").to_crs(METRIC_CRS).iloc[0]
    dx, dy = coords[:, 0] - center.x, coords[:, 1] - center.y
    dist_km = np.hypot(dx, dy) / 1000
    result["dist_center_km"] = dist_km
    result["ring"] = np.select([dist_km < RING_EDGES_KM[0], dist_km < RING_EDGES_KM[1]],
                               [RING_NAMES[0], RING_NAMES[1]], default=RING_NAMES[2])
    bearing = (np.degrees(np.arctan2(dx, dy)) + 360) % 360
    result["bearing_name"] = [BEARING_NAMES[int(((b + 22.5) % 360) // 45)] for b in bearing]

    flags = {short: _binary(gdf, col) for short, col in FLAG_FEATURES.items()}
    parts = {}
    for key in ["stops", "education", "health", "sport", "institution",
                "police", "mops", "cas"]:
        parts[key] = _nearest_utility(coords, flags[key])
    for key in ["cycle", "water", "toilets", "green"]:
        parts[key] = _local_mean(flags[key], adj)
    raw_cycle = pd.to_numeric(gdf.get("liczba_infrastruktury_rowerowej",
                                      pd.Series(np.nan, index=gdf.index)), errors="coerce").to_numpy(float)
    raw_cycle[raw_cycle < 0] = np.nan
    parts["cycle_amount"] = _robust_utility(_local_mean(np.log1p(raw_cycle), adj))

    # WENTYLACJA: włączamy dopiero po potwierdzeniu kierunku interpretacji.
    if ventilation_higher_is_better is not None:
        if not isinstance(ventilation_higher_is_better, bool):
            raise ValueError("ventilation_higher_is_better musi być True, False lub None")
        ventilation = pd.to_numeric(gdf.get("wentyl_potencjal",
                                 pd.Series(np.nan, index=gdf.index)), errors="coerce")
        parts["ventilation"] = _robust_utility(
            _local_mean(ventilation.to_numpy(float), adj),
            higher_is_better=ventilation_higher_is_better)
        INTERNAL_ENV = {"green": 0.60, "ventilation": 0.40}
    else:
        INTERNAL_ENV = {"green": 1.0}
    district = _districts_for_centers(geom, district_path, district_column)
    result["district_no"] = district
    prices = prepare_district_prices(price_path)
    result["price_m2_district"] = pd.Series(district).map(
        prices.set_index("district_no")["price_m2"] if not prices.empty
        else pd.Series(dtype=float)).to_numpy()
    parts["price_utility"] = _robust_utility(result["price_m2_district"], higher_is_better=False)

    # DOSTĘPNOŚĆ: punktowanie wyłącznie przy świadomym podaniu kierunku i skali.
    access = pd.to_numeric(gdf.get("score_niepelnosprawni",
                               pd.Series(np.nan, index=gdf.index)), errors="coerce").to_numpy(float)
    parts["access_utility"] = np.full(len(gdf), np.nan)
    if accessibility_higher_is_better is not None or accessibility_scale is not None:
        if not isinstance(accessibility_higher_is_better, bool) or accessibility_scale not in (1, 100):
            raise ValueError("Dla dostępności wymagane: accessibility_higher_is_better=True/False i accessibility_scale=1/100")
        valid = access[np.isfinite(access)]
        if len(valid) >= 20 and valid.min() >= 0 and valid.max() <= accessibility_scale:
            standardized = access / accessibility_scale
            if not accessibility_higher_is_better:
                standardized = 1 - standardized
            parts["access_utility"] = _local_mean(standardized, adj)
        else:
            warnings.warn("Nie można zastosować udokumentowanej skali dostępności — wymiar niedostępny")
    for key, values in parts.items():
        result[f"indicator_{key}"] = values
    for group in GROUPS:
        internal = INTERNAL_ENV if group == "Środowisko" else INTERNAL_WEIGHTS[group]
        result[group] = _combine(parts, internal)
        result[f"area_{group}"] = _local_mean(result[group], adj)

    for col in DESCRIPTIVE_FIELDS:
        if col in gdf:
            result[f"raw_{col}"] = gdf[col].to_numpy()
    result.attrs["model_notes"] = [
        "Odległości euklidesowe między centroidami, nie trasy piesze.",
        "Ceny: średnie ważone dla dzielnicy, tylko z oficjalnymi granicami.",
        "Hałas: bardzo słabe pokrycie; nie wchodzi do rankingu.",
        "Wentylacja: w raporcie, nie w rankingu bez walidacji definicji.",
        "MPZP i kategorie placówek: dane opisowe, nie arbitralne punkty.",
        "Brakujące dane nie są automatycznie traktowane jako zero.",
    ]
    return result


def score_city(features, weights, min_coverage=0.85,
               peer_normalize=False, sharpness=SHARPNESS):
    """Ważony MCDA/SAW na ocenach otoczenia; wynik 0–100 to indeks.

    ZMIANA: każdy wymiar jest zamieniany na percentyl (pct_<grupa>), zanim zostanie
    zsumowany. Dzięki temu:
      * nie ma remisów 'na suficie' (centrum nie wygrywa z automatu po GRID_ID),
      * wymiary mają porównywalną skalę, a suwaki naprawdę coś zmieniają.
    peer_normalize=True liczy percentyle wewnątrz strefy (Śródmieście/Miasto/Obrzeża),
    czyli ocenia okolicę na tle podobnych okolic, a nie całego Krakowa.
    """
    w = _weights(weights) ** float(sharpness)
    w = w / w.sum()
    missing_groups = [g for g, weight in zip(GROUPS, w) if weight > 0 and
                      not features[g].notna().any()]
    if missing_groups:
        raise ValueError("Brak danych dla wybranych wymiarów: " + ", ".join(missing_groups) +
                         ". Ceny wymagają oficjalnych granic dzielnic; "
                         "dostępność wymaga zatwierdzenia skali.")
    result = features.copy()
    group_key = result["ring"].to_numpy() if (peer_normalize and "ring" in result) else None
    for g in GROUPS:
        result[f"pct_{g}"] = _percentile(result[f"area_{g}"], group_key)
    local = result[GROUPS].to_numpy(float)
    pct = result[[f"pct_{g}" for g in GROUPS]].to_numpy(float)
    result["score"], result["coverage"] = _weighted(local, w)
    result["area_score"], result["area_coverage"] = _weighted(pct, w)
    result.loc[(result["area_coverage"] < min_coverage) |
               (result["coverage"] < min_coverage), "area_score"] = np.nan
    return result


def _eligible_positions(scored, area_radius_m, min_coverage):
    eligible = scored.loc[
        scored["area_score"].notna() &
        (scored["coverage"] >= min_coverage) &
        (scored["area_coverage"] >= min_coverage)
    ].copy()
    eligible = eligible.sort_values(["area_score", "GRID_ID"], ascending=[False, True])
    all_xy = scored[["x", "y"]].to_numpy(float)
    tree = cKDTree(all_xy)
    return eligible, tree


def select_places(scored, how_many=5, min_distance_m=1200,
                  area_radius_m=RADIUS_M, min_coverage=.85,
                  max_price_m2=None):
    """Klasyczna heurystyka TOP N (centra, nie granice spójnych skupisk)."""
    if how_many < 1 or min_distance_m <= 0 or area_radius_m <= 0:
        raise ValueError("Nieprawidłowe parametry selekcji")
    eligible, tree = _eligible_positions(scored, area_radius_m, min_coverage)
    if max_price_m2 is not None:
        if not np.isfinite(max_price_m2) or max_price_m2 <= 0:
            raise ValueError("Cena maksymalna musi być dodatnia")
        eligible = eligible.loc[eligible["price_m2_district"].notna() &
                                (eligible["price_m2_district"] <= max_price_m2)]
    selected = []
    chosen = []
    for _, row in eligible.iterrows():
        xy = np.array([row["x"], row["y"]], float)
        if any(np.linalg.norm(xy - prev) < min_distance_m for prev in chosen):
            continue
        near = scored.iloc[tree.query_ball_point(xy, area_radius_m)]
        valid = near["score"].dropna()
        if len(valid) < 3:
            continue
        item = row.to_dict()
        item.update({"area_n": len(near), "local_min_score": float(valid.min()),
                     "local_max_score": float(valid.max()),
                     "local_median_score": float(valid.median()),
                     "rank": len(selected) + 1})
        selected.append(item)
        chosen.append(xy)
        if len(selected) >= how_many:
            break
    return pd.DataFrame(selected), selected


def _signature(item, weights):
    """Wymiar, w którym miejsce wypada najlepiej (spośród aktywnych)."""
    best, val = None, -1.0
    for g in GROUPS:
        if weights is not None and float(weights.get(g, 0)) <= 0:
            continue
        v = item.get(f"pct_{g}", np.nan)
        if pd.notna(v) and float(v) > val:
            best, val = g, float(v)
    return best, val


def select_diverse_places(scored, how_many=5, min_distance_m=2000,
                          area_radius_m=RADIUS_M, min_coverage=.85,
                          cell_m=1500, diversity=.35, per_ring=False,
                          allowed_rings=None, weights=None):
    """NOWE: wybór TOP N, który nie zwraca pięciu sąsiednich punktów z jednego miejsca.

    1. Miasto dzielone jest na kafle (cell_m); z każdego bierzemy lokalnego 'championa'.
    2. Greedy: kolejne miejsce maksymalizuje mieszankę wyniku, odległości od już wybranych
       i różnicy PROFILU (inny charakter okolicy). diversity=0 -> czysty ranking.
    3. per_ring=True gwarantuje najpierw po jednym miejscu z każdej strefy.
    Zwraca (DataFrame, lista słowników) jak select_places.
    """
    if how_many < 1 or min_distance_m <= 0 or area_radius_m <= 0 or cell_m <= 0:
        raise ValueError("Nieprawidłowe parametry selekcji")
    if not 0 <= diversity <= 1:
        raise ValueError("diversity musi być w zakresie 0–1")
    eligible, tree = _eligible_positions(scored, area_radius_m, min_coverage)
    if allowed_rings:
        eligible = eligible.loc[eligible["ring"].isin(list(allowed_rings))]
    if eligible.empty:
        return pd.DataFrame(), []

    cells = (np.floor(eligible["x"] / cell_m).astype(int).astype(str) + "_" +
             np.floor(eligible["y"] / cell_m).astype(int).astype(str))
    champions = eligible.groupby(cells, sort=False).head(1)          # eligible jest już posortowane
    rest = eligible.drop(champions.index).head(600)
    pool = pd.concat([champions, rest]).reset_index(drop=True)

    active = [f"pct_{g}" for g in GROUPS
              if (weights is None or float(weights.get(g, 0)) > 0)
              and f"pct_{g}" in pool and pool[f"pct_{g}"].notna().any()]
    lo = float(pool["area_score"].quantile(.25))
    span = max(float(pool["area_score"].max()) - lo, 1e-9)
    base = ((pool["area_score"] - lo) / span).clip(0, 1).to_numpy(float)
    prof = np.nan_to_num(pool[active].to_numpy(float), nan=.5) if active else np.zeros((len(pool), 0))
    xy_all = pool[["x", "y"]].to_numpy(float)
    ring_arr = pool["ring"].to_numpy()
    records = pool.to_dict("records")

    alive = np.ones(len(pool), bool)
    chosen_idx: list[int] = []
    selected: list[dict] = []

    def value(i):
        if not chosen_idx:
            return base[i]
        d = np.linalg.norm(xy_all[chosen_idx] - xy_all[i], axis=1).min()
        if d < min_distance_m:
            return None
        geo = min(d / 6000.0, 1.0)
        pdist = np.abs(prof[chosen_idx] - prof[i]).mean(axis=1).min() if prof.shape[1] else 0.0
        return (1 - diversity) * base[i] + diversity * (.5 * geo + .5 * min(pdist / .35, 1.0))

    def add_best(mask):
        best, best_v = None, -1.0
        for i in np.flatnonzero(alive & mask):
            v = value(i)
            if v is not None and v > best_v:
                best, best_v = i, v
        if best is None:
            return False
        alive[best] = False
        near = scored.iloc[tree.query_ball_point(xy_all[best], area_radius_m)]
        valid = near["score"].dropna()
        if len(valid) < 3:
            return True                                   # kandydat odrzucony, szukamy dalej
        item = dict(records[best])
        sig, sig_val = _signature(item, weights)
        item.update({"area_n": len(near), "local_min_score": float(valid.min()),
                     "local_max_score": float(valid.max()),
                     "local_median_score": float(valid.median()),
                     "signature": sig, "signature_pct": sig_val})
        selected.append(item)
        chosen_idx.append(best)
        return True

    everyone = np.ones(len(pool), bool)
    if per_ring:
        for ring in RING_NAMES:
            if len(selected) >= how_many:
                break
            if (ring_arr == ring).any():
                while True:                                # pierwsze udane dodanie w strefie
                    before = len(selected)
                    ok = add_best(ring_arr == ring)
                    if not ok or len(selected) > before:
                        break
    while len(selected) < how_many:
        if not add_best(everyone):
            break

    if not selected:
        return pd.DataFrame(), []
    selected.sort(key=lambda r: -float(r["area_score"]))
    for k, item in enumerate(selected, 1):
        item["rank"] = k
    return pd.DataFrame(selected), selected


def explain_place(place, features, weights):
    """Wyjaśnia różnice OCEN OTOCZENIA względem mediany Krakowa."""
    output = []
    for group in GROUPS:
        column = f"area_{group}"
        value = place.get(column, np.nan)
        baseline = features[column].median()
        if weights.get(group, 0) > 0 and pd.notna(value) and pd.notna(baseline):
            output.append((group, round(100 * (float(value) - float(baseline)), 2)))
    return sorted(output, key=lambda p: p[1], reverse=True)


def pareto_comparison(top, weights):
    """Pareto jedynie wśród zaprezentowanych lokalizacji, na ocenach otoczenia."""
    if top.empty:
        return pd.DataFrame(columns=["rank", "pareto_efficient"])
    active = [f"area_{g}" for g in GROUPS if float(weights.get(g, 0)) > 0]
    values = top[active].to_numpy(float)
    efficient = []
    for i in range(len(values)):
        valid = np.isfinite(values[i])
        if not valid.all():
            efficient.append(pd.NA)
            continue
        dominated = any(j != i and np.isfinite(values[j]).all() and
                        np.all(values[j] >= values[i]) and np.any(values[j] > values[i])
                        for j in range(len(values)))
        efficient.append(not dominated)
    return pd.DataFrame({"rank": top["rank"].to_numpy(),
                         "pareto_efficient": pd.array(efficient, dtype="boolean")})


def sensitivity_analysis(features, weights, baseline_top, n_simulations=200,
                         perturbation=.20, min_distance_m=1200, seed=42,
                         min_coverage=.85, max_price_m2=None):
    """Monte Carlo: identyczna ścieżka score_city -> select_places jak w bazie.

    Losowe mnożniki wag z jednostajnego [1-p,1+p], dla wag >0.
    Trafienie jest dopasowaniem rekomendacji w promieniu 600 m.
    Częstość zależy od wybranego scenariusza zmian wag, NIE jest pewnością.
    """
    if baseline_top.empty:
        return pd.DataFrame(columns=["rank", "top5_frequency", "top1_frequency"])
    if n_simulations < 1 or not (0 <= perturbation < 1):
        raise ValueError("Nieprawidłowa liczba symulacji lub perturbacja")
    w0 = _weights(weights)
    active = w0 > 0
    rng = np.random.default_rng(seed)
    xy0 = baseline_top[["x", "y"]].to_numpy(float)
    hits = np.zeros(len(xy0), dtype=int)
    wins = np.zeros(len(xy0), dtype=int)
    for _ in range(n_simulations):
        w = w0.copy()
        w[active] *= rng.uniform(1 - perturbation, 1 + perturbation, active.sum())
        sampled = {g: float(val) for g, val in zip(GROUPS, w)}
        scored = score_city(features, sampled, min_coverage=min_coverage)
        top, _ = select_places(scored, how_many=len(xy0),
                               min_distance_m=min_distance_m,
                               min_coverage=min_coverage,
                               max_price_m2=max_price_m2)
        if top.empty:
            continue
        xy = top[["x", "y"]].to_numpy(float)
        distances = np.linalg.norm(xy0[:, None, :] - xy[None, :, :], axis=2)
        tolerance = min(600.0, min_distance_m / 2 - 1)
        hits += (distances.min(axis=1) <= tolerance)
        wins += (distances[:, 0] <= tolerance)
    return pd.DataFrame({
        "rank": baseline_top["rank"].to_numpy(),
        "top5_frequency": np.round(hits / n_simulations * 100, 1),
        "top1_frequency": np.round(wins / n_simulations * 100, 1),
    })


def location_report(place, features, price_table=None):
    """Opis lokalizacji wraz z jakością danych, bez imputowania braków."""
    summary = {"GRID_ID": place["GRID_ID"], "rank": place.get("rank"),
               "score": place.get("area_score"),
               "coverage": place.get("area_coverage"),
               "district_no": place.get("district_no"),
               "price_m2_district": place.get("price_m2_district")}
    for group in GROUPS:
        summary[group] = place.get(f"area_{group}", np.nan)
    for col in ["max_halas_dB", "max_halas_szyny_dB",
                "wentyl_potencjal", "wentyl_predkosc_v4m",
                "mpzp_przeznaczenia", "score_niepelnosprawni",
                "kategorie_edukacji", "rodzaje_sportu", "dyscypliny_sportowe"]:
        summary[col] = place.get(f"raw_{col}", np.nan)
    if price_table is not None and pd.notna(summary["district_no"]):
        row = price_table.loc[price_table["district_no"] == summary["district_no"]]
        if not row.empty:
            summary["price_n_locals"] = int(row.iloc[0]["n_locals"])
            summary["price_last_year"] = int(row.iloc[0]["last_year"])
    return summary


if __name__ == "__main__":
    print("Wczytywanie siatki...")
    features = prepare_city()
    print("Liczba heksagonów:", len(features))
    print(features[GROUPS].notna().mean().round(3).to_string())
    weights = DEFAULT_WEIGHTS.copy()
    scored = score_city(features, weights, peer_normalize=True)
    top, _ = select_diverse_places(scored, per_ring=True, weights=weights)
    print("\nTOP 5 (zróżnicowane):")
    if top.empty:
        print("Brak lokalizacji spełniających kryteria kompletności")
    else:
        print(top[["rank", "ring", "bearing_name", "dist_center_km", "signature",
                   "area_score", "lat", "lon"]].to_string(index=False))