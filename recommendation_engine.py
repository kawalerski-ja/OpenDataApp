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

SHARPNESS = 2.4



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





def _gaussian_adjacency(adjacency, coords, radius_m):
    """Gaussian kernel on the *existing* 500-m sparse neighborhood graph.

    Weights taper smoothly with distance, preventing a cell 499 m away from
    counting as strongly as the same cell's own observation. No network routes.
    """
    coo = adjacency.tocoo()
    delta = np.asarray(coords, float)[coo.row] - np.asarray(coords, float)[coo.col]
    dist2 = np.einsum("ij,ij->i", delta, delta)
    sigma = max(float(radius_m) / 2.0, 1.0)
    values = np.exp(-0.5 * dist2 / (sigma * sigma)).astype(np.float32)
    return csr_matrix((values, (coo.row, coo.col)), shape=adjacency.shape)


def _empirical_bayes_coverage(flag, adjacency, prior_strength=5.0):
    """Beta-binomial posterior mean of presence rate in adjacent hexes.

    A weak city-level prior stabilizes neighborhoods represented by just a few
    observed hexes; missing flags are excluded from the denominator.
    This is a prevalence of *hexes*, not a count of facilities.
    """
    flag = np.asarray(flag, float)
    valid = np.isfinite(flag)
    if not valid.any():
        return np.full(len(flag), np.nan)
    global_rate = float(np.mean(flag[valid]))
    denom = np.asarray(adjacency @ valid.astype(float)).ravel()
    successes = np.asarray(adjacency @ np.where(valid, flag, 0.0)).ravel()
    prior = float(prior_strength)
    out = (successes + prior * global_rate) / (denom + prior)
    out[denom == 0] = np.nan
    return out


def _weighted_information_adjustment(values, requested, strength=0.20):
    """Shrinkage-CRITIC adjustment for redundant criteria (unsupervised).

    Uses spread and correlation in the reference-city criterion matrix.
    User-selected zero weights stay exactly zero. Only 20% of the prior
    weight mass may move, so the *declared preferences* remain primary.
    """
    w = np.asarray(requested, float)
    active = np.flatnonzero(w > 0)
    if len(active) < 2:
        return w / w.sum()
    arr = np.asarray(values, float)[:, active]
    # Deterministic subsampling avoids expensive repeated analysis on 32k cells.
    if len(arr) > 8000:
        arr = arr[np.linspace(0, len(arr) - 1, 8000, dtype=int)]
    good = np.isfinite(arr).all(axis=1)
    arr = arr[good]
    if len(arr) < 30:
        return w / w.sum()
    dispersion = np.std(arr, axis=0)
    if np.count_nonzero(dispersion > 1e-12) < 2:
        return w / w.sum()
    corr = np.corrcoef(arr, rowvar=False)
    corr = np.nan_to_num(corr, nan=1.0)
    # Standard CRITIC: standard deviation × sum(1-correlation).
    # Negative correlation gives greater contrast; positive correlation is redundant.
    information = dispersion * np.maximum(0.0, 1.0 - np.clip(corr, -1.0, 1.0)).sum(axis=1)
    if information.sum() <= 1e-12:
        return w / w.sum()
    target = np.zeros_like(w)
    target[active] = information / information.sum()
    prior = w / w.sum()
    return (1.0 - strength) * prior + strength * target


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

    head = re.split(r"[,;\s-]", s)[0]

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
    spatial_kernel = _gaussian_adjacency(adj, coords, radius_m)

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

    # Bliskość i pokrycie terenu to różne pojęcia. Flaga 1 oznacza, że dany
    # heksagon ma usługę, a NIE liczbę fizycznych placówek. Udział heksagonów
    # z usługą traktujemy jako zastępczy wskaźnik jej rozpowszechnienia.
    for _key in ("stops", "education", "health", "sport", "institution", "police", "mops", "cas"):
        parts[_key + "_coverage"] = _empirical_bayes_coverage(flags[_key], spatial_kernel)
    for _key in ("cycle", "water", "toilets", "green"):
        parts[_key + "_nearest"] = _nearest_utility(coords, flags[_key], near_m=100, far_m=1200)

    raw_cycle = pd.to_numeric(gdf.get("liczba_infrastruktury_rowerowej",

                                      pd.Series(np.nan, index=gdf.index)), errors="coerce").to_numpy(float)

    raw_cycle[raw_cycle < 0] = np.nan

    parts["cycle_amount"] = _robust_utility(_local_mean(np.log1p(raw_cycle), spatial_kernel))



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

        # W analizie obszaru i rankingu używamy tej samej skali otoczenia.
        result[f"area_indicator_{key}"] = _local_mean(values, spatial_kernel)

    for group in GROUPS:

        internal = INTERNAL_ENV if group == "Środowisko" else INTERNAL_WEIGHTS[group]

        result[group] = _combine(parts, internal)

        result[f"area_{group}"] = _local_mean(result[group], spatial_kernel)



    for _key, _flag in flags.items():
        result[f"raw_flag_{_key}"] = _flag

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
        "Empiryczno-bayesowska estymacja udziału heksagonów z usługą; wygładzanie jądrowe.",
        "Korekta wag CRITIC (20%) penalizuje redundantne kryteria, bez zmiany zer preferencji.",

    ]

    return result





def score_city(features, weights, min_coverage=0.85,
               peer_normalize=False, sharpness=None, profile=None):
    """Profile-aware MCDA + tempered, data-driven CRITIC redundancy correction.

    Percentiles are relative to the city, not satisfaction probabilities.
    The correction takes at most 20% of criterion importance and cannot turn
    an excluded (zero-weight) criterion back on.
    """
    effective_profile = resolve_profile(weights, profile)
    w = _weights(weights)
    missing = [g for g, weight in zip(GROUPS, w) if weight > 0 and
               not features[g].notna().any()]
    if missing:
        raise ValueError("Brak danych dla wybranych wymiarów: " + ", ".join(missing))
    result = features.copy()
    local = profile_group_values(result, effective_profile, area=False)
    area = profile_group_values(result, effective_profile, area=True)
    grouping = result["ring"].to_numpy() if peer_normalize and "ring" in result else None
    for g in GROUPS:
        result[f"pct_{g}"] = _percentile(area[g], grouping)
        # Report the profile-adjusted category, not a misleading general average.
        result[g] = local[g]
        result[f"area_{g}"] = area[g]
    pcts = result[[f"pct_{g}" for g in GROUPS]].to_numpy(float)
    exponent = SHARPNESS if sharpness is None else float(sharpness)
    if exponent <= 0:
        raise ValueError("sharpness musi być dodatnie")
    subjective = _weights(weights) ** exponent
    subjective /= subjective.sum()
    adjusted = _weighted_information_adjustment(pcts, subjective, strength=.20)
    result["score"], result["coverage"] = _weighted(
        np.column_stack([local[g] for g in GROUPS]), adjusted)
    p = {g: result[f"pct_{g}"].to_numpy(float) for g in GROUPS}
    result["area_score"], result["area_coverage"] = _preference_score(
        p, weights, effective_profile, adjusted_weights=adjusted)
    result.loc[(result["area_coverage"] < min_coverage) |
               (result["coverage"] < min_coverage), "area_score"] = np.nan
    result.attrs["effective_weights"] = {g: float(v) for g, v in zip(GROUPS, adjusted)}
    result.attrs["profile"] = effective_profile
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

    fixed_profile = resolve_profile(weights, profile)
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

        scored = score_city(features, sampled, min_coverage=min_coverage, profile=fixed_profile)

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






# ---------------------------------------------------------------------------
# Moduł UI: ocena narysowanego obszaru, adresów i filtrowanie 18 dzielnic.
# Wszystkie odległości są geometryczne w EPSG:2178, NIE po sieci pieszej.
# ---------------------------------------------------------------------------
DISTRICTS = {
    1: "Stare Miasto", 2: "Grzegórzki", 3: "Prądnik Czerwony",
    4: "Prądnik Biały", 5: "Krowodrza", 6: "Bronowice",
    7: "Zwierzyniec", 8: "Dębniki", 9: "Łagiewniki-Borek Fałęcki",
    10: "Swoszowice", 11: "Podgórze Duchackie", 12: "Bieżanów-Prokocim",
    13: "Podgórze", 14: "Czyżyny", 15: "Mistrzejowice",
    16: "Bieńczyce", 17: "Wzgórza Krzesławickie", 18: "Nowa Huta",
}

USER_GROUPS = ["Transport", "Edukacja", "Zdrowie", "Sport", "Usługi publiczne", "Środowisko"]
PROFILE_WEIGHTS = {
    "Własne preferencje": {g: DEFAULT_WEIGHTS[g] for g in USER_GROUPS},
    "Rodzic": {"Transport": 3, "Edukacja": 5, "Zdrowie": 4,
               "Sport": 2, "Usługi publiczne": 1, "Środowisko": 4},
    "Senior": {"Transport": 5, "Edukacja": 0, "Zdrowie": 5,
               "Sport": 0, "Usługi publiczne": 5, "Środowisko": 1},
    "Student": {"Transport": 5, "Edukacja": 0, "Zdrowie": 1,
                "Sport": 4, "Usługi publiczne": 1, "Środowisko": 1},
    "Aktywny styl życia": {"Transport": 3, "Edukacja": 0, "Zdrowie": 0,
                            "Sport": 5, "Usługi publiczne": 1, "Środowisko": 5},
    "Bez samochodu": {"Transport": 5, "Edukacja": 1, "Zdrowie": 3,
                       "Sport": 2, "Usługi publiczne": 3, "Środowisko": 2},
    "Miłośnik zieleni": {"Transport": 1, "Edukacja": 1, "Zdrowie": 2,
                        "Sport": 2, "Usługi publiczne": 1, "Środowisko": 5},
}

# Profile dobierają też STRUKTURĘ składników kategorii.
# Części *_coverage to udział heksagonów z obiektem (nie liczba placówek),
# *_nearest to geometryczna odległość do najbliższego heksagonu z obiektem.
# WAGA 0 oznacza świadome pominięcie usługi, a nie brak danych.
BASE_COMPONENTS = {
    "Transport": {"stops": .48, "stops_coverage": .17,
                  "cycle": .12, "cycle_nearest": .08, "cycle_amount": .15},
    "Edukacja": {"education": .72, "education_coverage": .28},
    "Zdrowie": {"health": .82, "health_coverage": .18},
    "Sport": {"sport": .60, "sport_coverage": .40},
    "Usługi publiczne": {"institution": .28, "institution_coverage": .08,
                         "mops": .12, "cas": .12, "police": .05,
                         "water": .16, "toilets": .19},
    "Środowisko": {"green": .80, "green_nearest": .20},
}
PROFILE_INTERNAL = {
    "Rodzic": {
        "Transport": {"stops": .60, "stops_coverage": .20,
                      "cycle": .08, "cycle_nearest": .05, "cycle_amount": .07},
        "Edukacja": {"education": .86, "education_coverage": .14},
        "Zdrowie": {"health": .83, "health_coverage": .17},
        "Sport": {"sport": .65, "sport_coverage": .35},
        "Usługi publiczne": {"institution": .12, "water": .35,
                             "toilets": .43, "police": .10},
        "Środowisko": {"green": .80, "green_nearest": .20},
    },
    "Senior": {
        "Transport": {"stops": .88, "stops_coverage": .12},
        "Zdrowie": {"health": .90, "health_coverage": .10},
        "Usługi publiczne": {"cas": .52, "mops": .22,
                             "institution": .10, "toilets": .11, "water": .05},
    },
    "Student": {
        "Transport": {"stops": .55, "stops_coverage": .20,
                      "cycle": .12, "cycle_nearest": .07, "cycle_amount": .06},
        "Sport": {"sport": .50, "sport_coverage": .50},
        "Usługi publiczne": {"water": .35, "toilets": .35,
                             "institution": .30},
    },
    "Aktywny styl życia": {
        "Transport": {"stops": .08, "stops_coverage": .04,
                      "cycle": .36, "cycle_nearest": .20, "cycle_amount": .32},
        "Sport": {"sport": .45, "sport_coverage": .55},
        "Usługi publiczne": {"water": .55, "toilets": .35,
                             "institution": .10},
        "Środowisko": {"green": .62, "green_nearest": .38},
    },
    "Bez samochodu": {
        "Transport": {"stops": .86, "stops_coverage": .14},
        "Usługi publiczne": {"institution": .35, "water": .25,
                             "toilets": .25, "mops": .15},
    },
    "Miłośnik zieleni": {
        "Transport": {"stops": .25, "cycle": .25, "cycle_nearest": .20,
                      "cycle_amount": .30},
        "Środowisko": {"green": .88, "green_nearest": .12},
    },
}

# Kryteria, których niskie wyniki są bardziej problematyczne dla danego profilu.
PROFILE_ESSENTIAL = {
    "Rodzic": ("Edukacja", "Zdrowie", "Środowisko"),
    "Senior": ("Zdrowie", "Transport", "Usługi publiczne"),
    "Student": ("Transport", "Sport"),
    "Aktywny styl życia": ("Sport", "Środowisko"),
    "Bez samochodu": ("Transport", "Usługi publiczne"),
    "Miłośnik zieleni": ("Środowisko",),
}

def resolve_profile(weights, profile=None):
    """Wybór z UI albo zgodne z profilem wagi; ręczne suwaki = własne preferencje."""
    if profile in PROFILE_INTERNAL:
        return profile
    if profile in ("Własne preferencje", None):
        for name, preset in PROFILE_WEIGHTS.items():
            if name == "Własne preferencje":
                continue
            if all(abs(float(weights.get(g, 0)) - float(preset[g])) < 1e-9 for g in USER_GROUPS):
                return name
    return None


def profile_group_values(features, profile=None, area=False):
    """Wyniki grup w skali 0..1. Każdy profil wybiera też rodzaj usługi.

    Nie imputujemy brakujących wskaźników zerem. Dla starszych danych bez
    wskaźników dodatkowych stosujemy dostępne części zamiast powodować błąd.
    """
    prefix = "area_" if area else ""
    columns = {g: pd.to_numeric(features[f"{prefix}{g}"], errors="coerce").to_numpy(float)
               for g in GROUPS}
    for group in USER_GROUPS:
        components = PROFILE_INTERNAL.get(profile, {}).get(group, BASE_COMPONENTS[group])
        values, weights = [], []
        for key, weight in components.items():
            if weight <= 0:
                continue
            field = f"{prefix}indicator_{key}"
            if field in features:
                values.append(pd.to_numeric(features[field], errors="coerce").to_numpy(float))
                weights.append(weight)
        if values:
            w = np.asarray(weights, float)
            w /= w.sum()
            columns[group] = _weighted(np.column_stack(values), w)[0] / 100
    return columns



def profile_priority(weights, profile):
    if profile in PROFILE_ESSENTIAL:
        return [g for g in PROFILE_ESSENTIAL[profile] if float(weights.get(g, 0)) > 0]
    usable = sorted(((float(weights.get(g, 0)), g) for g in USER_GROUPS), reverse=True)
    maximum = usable[0][0] if usable else 0
    return [g for w, g in usable if maximum >= 4 and w >= maximum - 1 and w > 0]


def _preference_score(percentiles, weights, profile=None, adjusted_weights=None):
    """Ważona ocena + delikatna premia za brak bardzo słabego kluczowego kryterium.

    Soft-min jest w skali 0–1 i nie wymusza sztucznej różnorodności dzielnic.
    """
    if adjusted_weights is None:
        w = _weights(weights) ** SHARPNESS
        w /= w.sum()
    else:
        w = np.asarray(adjusted_weights, dtype=float)
    pct = np.column_stack([percentiles[g] for g in GROUPS])
    weighted, coverage = _weighted(pct, w)
    keys = profile_priority(weights, profile)
    if len(keys) >= 2:
        essential = np.column_stack([percentiles[k] for k in keys])
        valid = np.isfinite(essential)
        all_present = valid.all(axis=1)
        # Średnia geometryczna karze za istotne deficyty, lecz nie działa jak twardy próg.
        safe = np.maximum(np.where(valid, essential, 1.), 0.01)
        balance = 100 * np.exp(np.log(safe).mean(axis=1))
        weighted = np.where(all_present, .58 * weighted + .42 * balance, weighted)
    return weighted, coverage


def load_address_catalog(path=None):
    """Lokalny katalog a: adresy.parquet albo data/adresy.csv."""
    options = ([Path(path)] if path is not None else [
        ROOT / "adresy.parquet", ROOT / "data" / "adresy.parquet",
        ROOT / "data" / "adresy.csv", ROOT / "adresy.csv",
    ])
    target = next((p for p in options if p.is_file()), None)
    if target is None:
        raise FileNotFoundError("Brak adresy.parquet lub data/adresy.csv")
    df = pd.read_parquet(target) if target.suffix.lower() == ".parquet" else pd.read_csv(
        target, sep=";", encoding="utf-8-sig", low_memory=False)
    required = {"GRID_ID", "numer_dzie", "nazwa_ulic", "numer_adre"}
    if not required.issubset(df):
        raise ValueError("W katalogu adresów brakuje: " + ", ".join(sorted(required - set(df))))
    df = df[list(required)].copy()
    df["GRID_ID"] = df["GRID_ID"].astype("string").str.strip()
    df["numer_dzie"] = pd.to_numeric(df["numer_dzie"], errors="coerce")
    df = df.loc[df["GRID_ID"].notna() & df["numer_dzie"].isin(DISTRICTS)].copy()
    df["numer_dzie"] = df["numer_dzie"].astype(int)
    for col in ["nazwa_ulic", "numer_adre"]:
        df[col] = df[col].fillna("").astype(str).str.strip()
    df = df.loc[df["nazwa_ulic"].ne("") & df["numer_adre"].ne("")].copy()
    df["label"] = df["nazwa_ulic"] + " " + df["numer_adre"]
    df["district_name"] = df["numer_dzie"].map(DISTRICTS)
    return df.reset_index(drop=True)


def with_address_districts(features, addresses):
    """Przypisz dzielnicę z adresu, pozostałe ostrożnie z sąsiedztwa.

    Dane przy granicach mogą pozostać nierozstrzygnięte. To NIE zastępuje
    oficjalnego przebiegu granic i nie powinno być tak przedstawiane.
    """
    out = features.copy().reset_index(drop=True)
    counts = addresses.groupby(["GRID_ID", "numer_dzie"], observed=True).size().rename("n").reset_index()
    counts["share"] = counts["n"] / counts.groupby("GRID_ID")["n"].transform("sum")
    counts = counts.sort_values(["GRID_ID", "n", "numer_dzie"], ascending=[True, False, True])
    main = counts.drop_duplicates("GRID_ID").copy()
    main.loc[main["share"] < .75, "numer_dzie"] = np.nan
    out["district_no"] = out["GRID_ID"].map(main.set_index("GRID_ID")["numer_dzie"]).astype(float)
    out["district_source"] = np.where(out["district_no"].notna(), "adres w heksagonie", "nieustalona")
    xy = out[["x", "y"]].to_numpy(float)
    known = np.flatnonzero(out["district_no"].notna().to_numpy())
    missing = np.flatnonzero(out["district_no"].isna().to_numpy())
    if len(known) >= 6 and len(missing):
        dist, idx = cKDTree(xy[known]).query(xy[missing], k=6)
        labels = out["district_no"].to_numpy()[known][idx].astype(int)
        found = np.full(len(missing), np.nan)
        for i in range(len(missing)):
            if dist[i, 0] > 800 or dist[i, 2] > 1300:
                continue
            votes = 1 / (dist[i] + 100)
            vals = np.bincount(labels[i], weights=votes, minlength=19)
            best = int(vals.argmax())
            if best > 0 and vals[best] / vals.sum() >= .80 and (labels[i] == best).sum() >= 4:
                found[i] = best
        valid = np.isfinite(found)
        if valid.any():
            out.loc[missing[valid], "district_no"] = found[valid]
            out.loc[missing[valid], "district_source"] = "najbliższe adresy"
    out["district_name"] = out["district_no"].map(DISTRICTS)
    names = addresses.loc[addresses["nazwa_ulic"].str.len() > 1].groupby(
        ["GRID_ID", "nazwa_ulic"]).size().rename("n").reset_index().sort_values(
        ["GRID_ID", "n", "nazwa_ulic"], ascending=[True, False, True]).drop_duplicates("GRID_ID")
    out["street"] = out["GRID_ID"].map(names.set_index("GRID_ID")["nazwa_ulic"])
    return out


def hex_ids_from_selection(selection):
    """zapisuje DataFrame po usunięciu geometrii, klucz GRID_ID."""
    if selection is None:
        return []
    if isinstance(selection, pd.DataFrame):
        if "GRID_ID" not in selection.columns:
            raise ValueError("Zaznaczony obszar nie ma kolumny GRID_ID")
        values = selection["GRID_ID"].dropna().tolist()
    elif isinstance(selection, dict) and "GRID_ID" in selection:
        values = selection["GRID_ID"]
    elif isinstance(selection, (list, tuple, set, np.ndarray, pd.Series)):
        values = selection
    else:
        raise ValueError("Nieobsługiwany format obszaru. Oczekiwano DataFrame z GRID_ID.")
    return list(dict.fromkeys(str(v).strip() for v in values if pd.notna(v) and str(v).strip()))


def radius_grid_ids(features, grid_id, radius_m):
    """Wszystkie centroidy heksagonów <= zadany promień od centroidu adresu."""
    if radius_m < 0:
        raise ValueError("Promień nie może być ujemny")
    located = features.loc[features["GRID_ID"].eq(str(grid_id))]
    if located.empty:
        raise ValueError("Nie znaleziono adresu w siatce heksagonów")
    xy = features[["x", "y"]].to_numpy(float)
    center = located[["x", "y"]].iloc[0].to_numpy(float)
    idx = cKDTree(xy).query_ball_point(center, float(radius_m))
    return features.iloc[idx]["GRID_ID"].tolist()


def expand_selection_grid_ids(features, grid_ids, buffer_m=0):
    """Geometria przybliżona: ekspansja wokół centroidów wybranych heksagonów."""
    values = set(map(str, grid_ids))
    idx = np.flatnonzero(features["GRID_ID"].isin(values).to_numpy())
    if len(idx) == 0:
        return []
    if buffer_m <= 0:
        return features.iloc[idx]["GRID_ID"].tolist()
    xy = features[["x", "y"]].to_numpy(float)
    matches = cKDTree(xy).query_ball_point(xy[idx], float(buffer_m))
    ids = sorted({j for group in matches for j in group})
    return features.iloc[ids]["GRID_ID"].tolist()


def evaluate_grid_area(features, grid_ids, weights, min_coverage=.85, profile=None):
    """Ocena wybranego obszaru z profilem i faktycznym wpływem jego zasięgu.

    Oceniamy wszystkie heksagony faktycznie objęte wskazanym promieniem lub
    zaznaczeniem. Częściowa dostępność obiektów jest udziałem heksagonów
    posiadających dany obiekt, NIE ich bezpośrednią liczbą.

    Wskaźniki grup = połączenie: średniej profilu, zasięgu usług wewnątrz
    bufora oraz bliskości. Skala odniesienia to wybrane cechy miasta.
    """
    selected = features.loc[features["GRID_ID"].isin(set(map(str, grid_ids)))]
    if selected.empty:
        raise ValueError("Zaznaczenie nie obejmuje żadnego znanego heksagonu")
    effective_profile = resolve_profile(weights, profile)
    local = profile_group_values(selected, effective_profile, area=False)
    # Promień ma REALNY wpływ na ocenę: oceniamy zasięg faktycznie dostępnych
    # usług w obrębie bufora, a nie tylko uśredniamy uprzednio policzone 500 m.
    raw = {}
    components = PROFILE_INTERNAL.get(effective_profile, {})
    for group in USER_GROUPS:
        base = local[group]
        local_mean = float(np.nanmean(base)) if np.isfinite(base).any() else np.nan
        definitions = components.get(group, BASE_COMPONENTS[group])
        sources = []
        source_w = []
        for key, importance in definitions.items():
            if importance <= 0:
                continue
            # 'coverage' to odsetek heksagonów, nie liczba placówek.
            if key.endswith('_coverage'):
                flagkey = key[:-9]
                raw_name = FLAG_FEATURES.get(flagkey)
                if raw_name is not None and f'raw_flag_{flagkey}' in selected:
                    f = pd.to_numeric(selected[f'raw_flag_{flagkey}'], errors='coerce')
                    if f.notna().any():
                        # Nie traktujemy zera jako braku; średnia = udział heksów.
                        city_flag = pd.to_numeric(features[f'raw_flag_{flagkey}'], errors='coerce')
                        prior_p = float(city_flag.mean()) if city_flag.notna().any() else float(f.mean())
                        # Conjugate Beta posterior: 5 equivalent observed hexes.
                        val = float((f.sum() + 5.0 * prior_p) / (f.notna().sum() + 5.0))
                        sources.append(val)
                        source_w.append(importance)
                continue
            col = 'indicator_' + key
            if col in selected:
                arr = pd.to_numeric(selected[col], errors='coerce')
                if arr.notna().any():
                    # Dla odległości: interesuje nas bliskość najlepszej dostępnej
                    # usługi, lecz większy bufor nie daje bezwarunkowej premii.
                    if key in ('stops','education','health','sport','institution',
                               'police','mops','cas','green_nearest','cycle_nearest'):
                        val = float(np.nanpercentile(arr, 75))
                    else:
                        val = float(arr.mean())
                    sources.append(val)
                    source_w.append(importance)
        # Uwzględniamy też rzeczywistą, zmieniającą się z promieniem strukturę
        # heksagonów (nie tylko najbliższy obiekt).
        if sources:
            component_score = float(np.average(sources, weights=source_w))
            raw[group] = float(.55 * component_score + .45 * local_mean) if np.isfinite(local_mean) else component_score
        else:
            raw[group] = local_mean
    # Rozkład referencyjny miasta liczony w tej samej przestrzeni cech;
    # porównanie ma charakter względny i nie udaje procentu potrzeb spełnionych.
    reference_values = profile_group_values(features, effective_profile, area=True)
    percentile = {}
    for group, value in raw.items():
        ref = np.sort(reference_values[group][np.isfinite(reference_values[group])])
        if not np.isfinite(value) or len(ref) == 0:
            percentile[group] = np.nan
            continue
        lo = np.searchsorted(ref, value, side='left')
        hi = np.searchsorted(ref, value, side='right')
        percentile[group] = float(np.clip((lo + hi) / (2 * len(ref)), 0., 1.))
    inputs = {g: np.array([percentile.get(g, np.nan)]) for g in GROUPS}
    # Same CRITIC adjustment as city-wide ranking, calculated on the reference.
    reference_percentiles = np.column_stack([
        _percentile(reference_values[g]) for g in GROUPS])
    prior = _weights(weights) ** SHARPNESS
    prior /= prior.sum()
    corrected = _weighted_information_adjustment(reference_percentiles, prior, strength=.20)
    score, coverage = _preference_score(
        inputs, weights, effective_profile, adjusted_weights=corrected)
    covered = float(coverage[0])
    return {'score': float(score[0]) if covered >= min_coverage else np.nan,
            'coverage': covered, 'cells': len(selected),
            'criteria': percentile, 'raw_criteria': raw,
            'frame': selected,
            'lat': float(selected['lat'].mean()),
            'lon': float(selected['lon'].mean())}



def select_places_in_districts(scored, district_names, count=5, min_distance_m=1200):
    """TOP 5: wyraźne preferencje + zróżnicowanie przestrzenne bez wymuszania dzielnic.

    W każdym kroku spośród kandydatów o wyniku do 3 pkt poniżej najlepszego
    wybieramy najmniej podobne przestrzennie miejsce. Nie obniżamy oceny MCDA.
    """
    pool = scored.loc[scored["district_name"].isin(set(district_names))].copy()
    eligible = pool.loc[pool["area_score"].notna() & pool["coverage"].ge(.85)
                        & pool["area_coverage"].ge(.85)].sort_values(
        ["area_score", "GRID_ID"], ascending=[False, True]).copy()
    if eligible.empty:
        return pd.DataFrame()
    points = eligible[["x", "y"]].to_numpy(float)
    all_tree = cKDTree(scored[["x", "y"]].to_numpy(float))
    chosen, chosen_xy = [], []
    remaining = np.ones(len(eligible), dtype=bool)
    for _ in range(count):
        distances = (np.linalg.norm(points[:, None, :] - np.stack(chosen_xy)[None, :, :], axis=2).min(axis=1)
                     if chosen_xy else np.full(len(eligible), np.inf))
        possible = np.flatnonzero(remaining & (distances >= min_distance_m))
        if not len(possible):
            break
        scores = eligible.iloc[possible]["area_score"].to_numpy(float)
        high = float(np.nanmax(scores))
        competitive = possible[scores >= high - 3.0]
        if chosen_xy and len(competitive) > 1:
            best = competitive[np.argmax(np.minimum(distances[competitive], 5000))]
        else:
            best = possible[int(np.argmax(scores))]
        remaining[best] = False
        row = eligible.iloc[best]
        near = scored.iloc[all_tree.query_ball_point(points[best], 500)]
        if near["score"].notna().sum() < 3:
            continue
        item = row.to_dict()
        item["rank"] = len(chosen) + 1
        chosen.append(item)
        chosen_xy.append(points[best])
    return pd.DataFrame(chosen)


def spatial_morans_i(scored, column="area_score", k=6, permutations=199,
                     seed=42, max_points=6000):
    """Moran's I on a deterministic spatial sample, with randomization p-value.

    A spatial *diagnostic* of clustering, not a predictor or a score booster.
    Null: the score is exchangeable among the sampled centroids. The random
    sample and kNN graph matter; this is not an inferential sample of residents.
    """
    data = scored.loc[np.isfinite(pd.to_numeric(scored[column], errors="coerce")),
                      ["x", "y", column]].copy()
    if len(data) < max(k + 3, 20):
        raise ValueError("Za mało obserwacji do statystyki Morana")
    if len(data) > max_points:
        idx = np.linspace(0, len(data)-1, max_points, dtype=int)
        data = data.iloc[idx].copy()
    coords = data[["x", "y"]].to_numpy(float)
    values = data[column].to_numpy(float)
    n = len(values)
    k = min(int(k), n - 1)
    neighbors = cKDTree(coords).query(coords, k=k+1)[1][:, 1:]
    z = values - values.mean()
    denom = np.dot(z, z)
    if denom <= 1e-12:
        raise ValueError("Brak zróżnicowania wskaźnika")
    def statistic(v):
        return float(np.sum(v[:, None] * v[neighbors]) / (k * np.dot(v,v)))
    observed = statistic(z)
    if permutations <= 0:
        return {"morans_i": observed, "permutation_p": np.nan,
                "n": n, "k": k, "permutations": 0}
    rng = np.random.default_rng(seed)
    null = np.array([statistic(rng.permutation(z)) for _ in range(permutations)])
    expected = -1 / (n - 1)
    pval = (1 + np.count_nonzero(np.abs(null - expected) >= abs(observed - expected))) / (permutations + 1)
    return {"morans_i": observed, "permutation_p": float(pval),
            "n": n, "k": k, "permutations": int(permutations)}


def profile_overlap_report(features, profiles=None, top_n=100):
    """Reproducible diagnostic: how differently profiles rank the same city.

    Jaccard similarity between top-N hex sets; lower indicates a real change
    in preferred locations. This does not alter recommendations or the UI.
    """
    profiles = profiles or ["Rodzic", "Senior", "Student", "Aktywny styl życia",
                             "Bez samochodu", "Miłośnik zieleni"]
    picks = {}
    for name in profiles:
        result = score_city(features, PROFILE_WEIGHTS[name], profile=name)
        picks[name] = set(result.nlargest(top_n, "area_score")["GRID_ID"].astype(str))
    records = []
    for i, a in enumerate(profiles):
        for b in profiles[i+1:]:
            common = len(picks[a] & picks[b])
            all_ = len(picks[a] | picks[b])
            records.append({"profil_a": a, "profil_b": b,
                            "wspolne_top_n": common,
                            "jaccard": common / all_ if all_ else np.nan})
    return pd.DataFrame(records)


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