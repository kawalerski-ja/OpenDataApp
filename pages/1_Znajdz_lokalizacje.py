"""Prosty wybór lokalizacji w 18 dzielnicach Krakowa.

Wymaga plików w katalogu projektu:
  recommendation_engine.py, krakow_master_grid_komplet.parquet,
  adresy.parquet (lub data/adresy.csv).
Nie zmienia app.py, mapy Jana ani danych wejściowych.
"""
from __future__ import annotations

import html
import json
import re
import sys
import time
import unicodedata
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from scipy.spatial import cKDTree
from shapely.geometry import Point

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recommendation_engine import (  # noqa: E402
    DEFAULT_WEIGHTS, GROUPS, prepare_city, score_city,
)

st.set_page_config(page_title="Znajdź lokalizację w Krakowie", page_icon="📍", layout="wide")

# Oficjalne dzielnice administracyjne (nie historyczne dzielnice katastralne).
# Wyświetlamy tylko NAZWY, bez „Dzielnica VI”, numerów itp.
DISTRICTS = {
    1: "Stare Miasto", 2: "Grzegórzki", 3: "Prądnik Czerwony",
    4: "Prądnik Biały", 5: "Krowodrza", 6: "Bronowice",
    7: "Zwierzyniec", 8: "Dębniki", 9: "Łagiewniki-Borek Fałęcki",
    10: "Swoszowice", 11: "Podgórze Duchackie", 12: "Bieżanów-Prokocim",
    13: "Podgórze", 14: "Czyżyny", 15: "Mistrzejowice",
    16: "Bieńczyce", 17: "Wzgórza Krzesławickie", 18: "Nowa Huta",
}
NAME_TO_NUMBER = {name: number for number, name in DISTRICTS.items()}
LABELS = {
    "Transport": "Komunikacja miejska i rower",
    "Edukacja": "Szkoły i edukacja",
    "Zdrowie": "Przychodnie i opieka zdrowotna",
    "Sport": "Sport i rekreacja",
    "Usługi publiczne": "Usługi w okolicy",
    "Środowisko": "Zielone korytarze",
}
ACTIVE = list(LABELS)

st.markdown("""
<style>
.block-container {max-width: 1420px; padding-top: 1.3rem; padding-bottom: 2.5rem}
.hero {background:linear-gradient(110deg,#143e4a,#147c75);color:white;
       border-radius:18px;padding:1.6rem 2rem;margin-bottom:1rem}
.hero h1 {color:white!important;margin:0 0 .4rem 0;font-size:2rem}
.hero p {margin:0;color:#e4f5f1}
div[data-testid="stButton"] button {border-radius:9px}
</style>
""", unsafe_allow_html=True)
st.markdown("""
<div class="hero"><h1>📍 Znajdź swoje miejsce w Krakowie</h1>
<p>Wybierz dzielnice i to, na czym Ci zależy. Znajdziemy pięć miejsc na mapie.</p></div>
""", unsafe_allow_html=True)


def simplify(value):
    """Uproszczenie nazwy ulicy podczas lokalnego wyszukiwania."""
    t = unicodedata.normalize("NFKD", str(value).lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = re.sub(r"\b(ulica|ul\.|ul)\b", " ", t)
    return re.sub(r"[^a-z0-9]+", " ", t).strip()


@st.cache_data(show_spinner="Wczytujemy dane Krakowa…")
def load_features():
    return prepare_city()


@st.cache_data(show_spinner="Wczytujemy adresy i dzielnice…")
def load_addresses():
    paths = [ROOT / "adresy.parquet", ROOT / "data" / "adresy.parquet",
             ROOT / "data" / "adresy.csv", ROOT / "adresy.csv"]
    path = next((p for p in paths if p.is_file()), None)
    if path is None:
        raise FileNotFoundError(
            "Brak danych adresowych. Umieść adresy.parquet w OpenDataApp/ "
            "albo adresy.csv w OpenDataApp/data/.")
    if path.suffix == ".parquet":
        addresses = pd.read_parquet(path)
    else:
        addresses = pd.read_csv(path, sep=";", encoding="utf-8-sig", low_memory=False)
    needed = {"GRID_ID", "numer_dzie", "nazwa_ulic", "numer_adre"}
    if not needed.issubset(addresses.columns):
        raise ValueError("Brak kolumn adresowych: " + ", ".join(sorted(needed - set(addresses.columns))))
    out = addresses[["GRID_ID", "numer_dzie", "nazwa_ulic", "numer_adre"]].copy()
    out["GRID_ID"] = out["GRID_ID"].astype("string").str.strip()
    out["numer_dzie"] = pd.to_numeric(out["numer_dzie"], errors="coerce")
    out = out[out["GRID_ID"].notna() & out["numer_dzie"].isin(DISTRICTS)].copy()
    out["numer_dzie"] = out["numer_dzie"].astype(int)
    out["nazwa_ulic"] = out["nazwa_ulic"].fillna("").astype(str).str.strip()
    out["numer_adre"] = out["numer_adre"].fillna("").astype(str).str.strip()
    return out


@st.cache_data(show_spinner="Przypisujemy heksagony do dzielnic…")
def attach_districts(features, addresses):
    """Najpierw etykieta z adresów w heksagonie, potem ostrożne najbliższe sąsiedztwo.

    Niepewne punkty przy granicach zostają bez przypisania. Bez geometrii
    granic administracyjnych nie da się zagwarantować 100% poprawności.
    """
    counts = (addresses.groupby(["GRID_ID", "numer_dzie"]).size()
              .rename("n").reset_index())
    totals = counts.groupby("GRID_ID")["n"].transform("sum")
    counts["share"] = counts["n"] / totals
    counts = counts.sort_values(["GRID_ID", "n", "numer_dzie"],
                                ascending=[True, False, True]).drop_duplicates("GRID_ID")
    counts.loc[counts["share"] < .75, "numer_dzie"] = np.nan
    by_hex = counts.set_index("GRID_ID")["numer_dzie"]
    result = features.copy()
    result["district_no"] = result["GRID_ID"].map(by_hex).astype(float)
    result["district_source"] = np.where(result["district_no"].notna(), "adres w heksagonie", "nieustalona")

    xy = result[["x", "y"]].to_numpy(dtype=float)
    known_mask = result["district_no"].notna().to_numpy()
    missing_idx = np.flatnonzero(~known_mask)
    if known_mask.sum() >= 6 and len(missing_idx):
        ref = np.flatnonzero(known_mask)
        tree = cKDTree(xy[ref])
        distances, neighbors = tree.query(xy[missing_idx], k=6)
        labels = result["district_no"].to_numpy()[ref][neighbors].astype(int)
        for j, pos in enumerate(missing_idx):
            ds, labs = distances[j], labels[j]
            # Bliskość jest konieczna; 6 punktów z jednej strony granicy nie
            # powinno dominować, jeśli najbliższy adres jest daleko.
            if ds[0] > 800 or ds[2] > 1300:
                continue
            votes = 1.0 / (ds + 100.0)
            shares = {int(d): float(votes[labs == d].sum()) for d in np.unique(labs)}
            winner = max(shares, key=shares.get)
            confidence = shares[winner] / sum(shares.values())
            if confidence >= .80 and int((labs == winner).sum()) >= 4:
                result.at[pos, "district_no"] = float(winner)
                result.at[pos, "district_source"] = "najbliższe adresy"

    result["district_name"] = result["district_no"].map(DISTRICTS)
    # Ulica do opisów: rzeczywista nazwa z danego heksagonu; nie wymyślamy adresu.
    valid_names = addresses[addresses["nazwa_ulic"].str.len() > 1]
    popular = (valid_names.groupby(["GRID_ID", "nazwa_ulic"]).size()
               .rename("n").reset_index()
               .sort_values(["GRID_ID", "n", "nazwa_ulic"], ascending=[True, False, True])
               .drop_duplicates("GRID_ID"))
    result["street"] = result["GRID_ID"].map(popular.set_index("GRID_ID")["nazwa_ulic"])
    return result


def score_label(value):
    return f"{float(value):.0f}/100" if pd.notna(value) else "brak oceny"


def place_name(row):
    district = row.get("district_name")
    street = row.get("street")
    if pd.notna(street) and str(street).strip():
        return f"okolice ul. {street} · {district}"
    return f"{district} · okolica wskazana na mapie"


def describe(row, features, weights):
    differences = []
    for group in ACTIVE:
        if weights[group] <= 0:
            continue
        value = row.get(f"area_{group}", np.nan)
        median = features[f"area_{group}"].median()
        if pd.notna(value) and pd.notna(median):
            differences.append((float(value - median), group))
    if not differences:
        return "Brak wystarczających danych do opisu."
    differences.sort(reverse=True)
    best = differences[0]
    worst = differences[-1]
    good = (f"Atutem jest {LABELS[best[1]].lower()}." if best[0] > 0
            else "Miejsce odpowiada wybranym priorytetom, choć bez wyraźnej przewagi nad miastem.")
    caution = (f"Słabszą stroną jest {LABELS[worst[1]].lower()}." if worst[0] < 0 else
               "W wybranych kryteriach nie wypada poniżej mediany Krakowa.")
    return good + " " + caution


def choose_places(scored, selected_districts, how_many=5, min_distance_m=1200):
    """TOP 5 z wybranych dzielnic; wyniki otoczenia policzone dla całego miasta."""
    options = scored.loc[
        scored["district_name"].isin(selected_districts)
        & scored["area_score"].notna()
        & scored["coverage"].ge(.85)
        & scored["area_coverage"].ge(.85)
    ].sort_values(["area_score", "GRID_ID"], ascending=[False, True])
    all_xy = scored[["x", "y"]].to_numpy(float)
    tree = cKDTree(all_xy)
    chosen = []
    centers = []
    for _, row in options.iterrows():
        xy = np.array([row["x"], row["y"]], float)
        if any(np.linalg.norm(xy - prev) < min_distance_m for prev in centers):
            continue
        nearby = scored.iloc[tree.query_ball_point(xy, 500)]
        if nearby["score"].notna().sum() < 3:
            continue
        item = row.to_dict()
        item["rank"] = len(chosen) + 1
        chosen.append(item)
        centers.append(xy)
        if len(chosen) >= how_many:
            break
    return pd.DataFrame(chosen)


def build_map(top, selected=0):
    m = folium.Map(location=[float(top["lat"].mean()), float(top["lon"].mean())],
                   zoom_start=12, tiles="OpenStreetMap", control_scale=True)
    bounds = []
    for i, row in top.reset_index(drop=True).iterrows():
        lat, lon = float(row["lat"]), float(row["lon"])
        bounds.append((lat, lon))
        color = "#0C8276" if i == selected else "#2976AB"
        title = html.escape(place_name(row))
        folium.Circle((lat, lon), radius=500, color=color, fill=True,
                      fill_opacity=.12 if i == selected else .04).add_to(m)
        folium.Marker(
            [lat, lon], tooltip=f"{i + 1}. {title}",
            popup=folium.Popup(f"<b>{title}</b><div>Ocena: {score_label(row['area_score'])}</div>",
                               max_width=280),
            icon=folium.DivIcon(icon_size=(34, 34), icon_anchor=(17, 17), html=(
                f'<div style="background:{color};color:white;border:3px solid white;'
                'width:34px;height:34px;border-radius:50%;display:flex;'
                'align-items:center;justify-content:center;box-shadow:0 2px 7px #5558;'
                f'font-weight:700">{i+1}</div>')),
        ).add_to(m)
    if len(bounds) > 1:
        m.fit_bounds(bounds, padding=(55, 55))
    return m.get_root().render()


def nearest_city_hex(features, latitude, longitude):
    point = gpd.GeoSeries([Point(longitude, latitude)], crs="EPSG:4326").to_crs("EPSG:2178").iloc[0]
    distance, index = cKDTree(features[["x", "y"]].to_numpy(float)).query([point.x, point.y])
    if distance > 1000:
        raise ValueError("Adres jest poza obszarem obsługiwanym przez aplikację.")
    return int(index)


@st.cache_data(ttl=86400, show_spinner=False)
def geocode_address(text):
    params = urlencode({"q": text.strip() + ", Kraków, Polska", "format": "jsonv2",
                        "limit": 1, "countrycodes": "pl"})
    request = Request("https://nominatim.openstreetmap.org/search?" + params,
                      headers={"User-Agent": "KrakowOpenDataApp/1.0 (github.com/kawalerski-ja/OpenDataApp)",
                               "Accept": "application/json"})
    try:
        with urlopen(request, timeout=10) as response:
            records = json.load(response)
    except (HTTPError, URLError, TimeoutError) as exc:
        raise ValueError("Nie udało się znaleźć adresu online.") from exc
    if not records:
        raise ValueError("Nie znaleziono podanego adresu.")
    lat, lon = float(records[0]["lat"]), float(records[0]["lon"])
    if not (49.9 <= lat <= 50.2 and 19.7 <= lon <= 20.2):
        raise ValueError("Znaleziony adres nie znajduje się w Krakowie.")
    return lat, lon


def match_local_address(query, addresses, features):
    """Dopasowanie ulicy/adresu z miejskiego pliku, bez internetu.

    Pełny numer budynku preferowany. Przy samej nazwie ulicy, jeżeli
    ulicę reprezentuje wiele heksagonów, wynik jest niejednoznaczny.
    """
    cleaned = simplify(query)
    if not cleaned:
        raise ValueError("Wpisz nazwę ulicy lub adres.")
    # Usuń numer budynku tylko na końcu zapytania, np. „Długa 5A”.
    match = re.match(r"^(.*?)\s+(\d+[a-z]?)$", cleaned)
    street, number = (match.group(1), match.group(2)) if match else (cleaned, None)
    sub = addresses.copy()
    sub["street_key"] = sub["nazwa_ulic"].map(simplify)
    sub = sub[sub["street_key"] == street]
    if number:
        sub = sub[sub["numer_adre"].map(simplify) == number]
    grids = sub["GRID_ID"].dropna().unique()
    valid = features.loc[features["GRID_ID"].isin(grids)]
    if len(valid) == 1:
        return int(valid.index[0])
    return None


# -------------------- I. KRYTERIA I DZIELNICE --------------------
st.subheader("1. Gdzie i czego szukasz?")
with st.form("preferences"):
    all_names = list(DISTRICTS.values())
    chosen_names = st.multiselect(
        "Wybierz dzielnice (możesz zaznaczyć kilka)",
        options=all_names, default=all_names,
        help="Wszystkie 18 dzielnic są dostępne pod jedną nazwą; np. Bronowice występują tylko raz.",
    )
    st.caption("Jak bardzo liczą się dla Ciebie poniższe rzeczy? 0 — bez znaczenia, 5 — najważniejsze.")
    left, right = st.columns(2, gap="large")
    w = {}
    for i, group in enumerate(ACTIVE):
        with (left if i % 2 == 0 else right):
            w[group] = st.slider(LABELS[group], 0, 5, int(DEFAULT_WEIGHTS[group]))
    submit = st.form_submit_button("🔎 Znajdź pięć miejsc", type="primary", use_container_width=True)

if "search_config" not in st.session_state:
    st.session_state.search_config = (all_names, {g: int(DEFAULT_WEIGHTS[g]) for g in ACTIVE})
if submit:
    if not chosen_names:
        st.warning("Wybierz przynajmniej jedną dzielnicę.")
        st.stop()
    if not any(w.values()):
        st.warning("Ustaw co najmniej jeden suwak powyżej zera.")
        st.stop()
    st.session_state.search_config = (chosen_names, w)
    st.session_state.selected_place = 0

chosen_names, applied = st.session_state.search_config
weights = {g: 0 for g in GROUPS}
weights.update(applied)

try:
    features = load_features()
    addresses = load_addresses()
    features = attach_districts(features, addresses)
    scored = score_city(features, weights)
    top = choose_places(scored, chosen_names)
except Exception as exc:
    st.error("Nie udało się przygotować rekomendacji. Sprawdź plik adresowy i silnik analityczny.")
    st.exception(exc)
    st.stop()

if top.empty:
    st.warning("Nie znaleziono rekomendacji w wybranych dzielnicach. Zaznacz więcej dzielnic.")
    st.stop()

# -------------------- II. MAPA I OPISY --------------------
st.divider()
st.subheader(f"2. {len(top)} miejsc dopasowanych do Twoich potrzeb")
selected = int(st.session_state.get("selected_place", 0))
selected = min(selected, len(top) - 1)
map_col, text_col = st.columns([1.7, 1], gap="large")
with map_col:
    components.html(build_map(top, selected), height=590, scrolling=False)
    st.caption("Punkty oznaczają środki analizowanych obszarów, okręgi — otoczenie 500 m."
               " Podkład mapy wymaga internetu.")
    with st.expander("Mapa nie wyświetla się?"):
        st.map(top[["lat", "lon"]], latitude="lat", longitude="lon", zoom=11)
with text_col:
    for i, row in top.iterrows():
        with st.container(border=True):
            st.markdown(f"**{i+1}. {place_name(row)}**")
            st.caption(f"Dopasowanie: {score_label(row['area_score'])}")
            st.write(describe(row, features, weights))
            if st.button("Pokaż na mapie" if i != selected else "✓ Wybrane miejsce",
                         key=f"place_{i}", use_container_width=True):
                st.session_state.selected_place = i
                st.rerun()

# -------------------- III. PORÓWNANIE --------------------
st.divider()
st.subheader("3. Porównaj dwie lokalizacje")
mode = st.radio("Co porównać?", ["Dwie propozycje z mapy", "Dwa adresy w Krakowie"], horizontal=True)
compare = None
if mode == "Dwie propozycje z mapy":
    l, r = st.columns(2)
    with l:
        a = st.selectbox("Pierwsze miejsce", range(len(top)), format_func=lambda i: place_name(top.iloc[i]))
    with r:
        b = st.selectbox("Drugie miejsce", range(len(top)), index=min(1, len(top)-1),
                         format_func=lambda i: place_name(top.iloc[i]))
    if a != b:
        compare = (top.iloc[a], top.iloc[b], place_name(top.iloc[a]), place_name(top.iloc[b]))
else:
    with st.form("addresses"):
        l, r = st.columns(2)
        with l:
            text_a = st.text_input("Pierwsza ulica i numer", placeholder="np. Bronowicka 104")
        with r:
            text_b = st.text_input("Druga ulica i numer", placeholder="np. Zarzecze 73")
        do_compare = st.form_submit_button("Porównaj adresy", type="primary")
    if do_compare:
        try:
            ids = []
            for i, text in enumerate((text_a, text_b)):
                if not text.strip():
                    raise ValueError("Uzupełnij obydwa adresy.")
                idx = match_local_address(text, addresses, scored)
                if idx is None:
                    lat, lon = geocode_address(text)
                    idx = nearest_city_hex(scored, lat, lon)
                    if i == 0:
                        time.sleep(1.1)  # Nominatim: nie więcej niż 1 zapytanie na sekundę
                ids.append(idx)
            st.session_state.address_compare = (text_a, text_b, *ids)
        except ValueError as exc:
            st.session_state.pop("address_compare", None)
            st.warning(str(exc))
    if "address_compare" in st.session_state:
        text_a, text_b, ia, ib = st.session_state.address_compare
        compare = (scored.iloc[ia], scored.iloc[ib], text_a, text_b)
        st.caption("Adresy są przypisywane do heksagonów, a nie oceniane jako pojedyncze budynki.")

if compare:
    a, b, label_a, label_b = compare
    st.write(f"**{label_a}:** {score_label(a['area_score'])}  |  "
             f"**{label_b}:** {score_label(b['area_score'])}")
    rows = []
    for group in ACTIVE:
        if weights[group] == 0:
            continue
        av, bv = a.get(f"area_{group}", np.nan), b.get(f"area_{group}", np.nan)
        rows.append({"Co porównujemy": LABELS[group],
                     "Pierwsze miejsce": round(float(av) * 100) if pd.notna(av) else None,
                     "Drugie miejsce": round(float(bv) * 100) if pd.notna(bv) else None})
    if rows:
        st.dataframe(pd.DataFrame(rows).set_index("Co porównujemy"), use_container_width=True)
        st.caption("Oceny kryteriów są wskaźnikami modelu w skali 0–100, nie oceną budynku.")

unresolved = int(features["district_no"].isna().sum())
if unresolved:
    st.caption(f"Uwaga: dla {unresolved:,} heksagonów nie udało się wiarygodnie określić "
               "dzielnicy z punktów adresowych. Nie są używane jako kandydaci w rankingu dzielnicowym.")
st.caption("Dane o cenach i dostępności dla osób z niepełnosprawnościami nie są obecnie "
           "uwzględniane w punktacji. Granice dzielnic są tu przybliżane z danych adresowych.")
