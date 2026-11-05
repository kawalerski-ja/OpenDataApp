"""Moje 15 minut — samodzielny interfejs ocen i rekomendacji Krakowa.

Moduł integracji: importuj render_location_panel z app.py.
Nie modyfikuje mapy ani plików danych.
"""
from __future__ import annotations

import html
import sys
import unicodedata
import re
from pathlib import Path

import folium
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parent
if not (ROOT / "recommendation_engine.py").exists():
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recommendation_engine import (  # noqa: E402
    DEFAULT_WEIGHTS, DISTRICTS, GROUPS, USER_GROUPS, PROFILE_WEIGHTS,
    prepare_city, load_address_catalog, with_address_districts,
    hex_ids_from_selection, evaluate_grid_area, score_city, select_places_in_districts,
)


LABELS = {
    "Transport": "Komunikacja i rower",
    "Edukacja": "Edukacja",
    "Zdrowie": "Opieka zdrowotna",
    "Sport": "Sport i rekreacja",
    "Usługi publiczne": "Usługi w okolicy",
    "Środowisko": "Zielone korytarze",
}


@st.cache_resource(show_spinner="Przygotowujemy dane Krakowa…")
def city_data():
    base = prepare_city()
    addr = load_address_catalog()
    enriched = with_address_districts(base, addr)
    lookup = addr.drop_duplicates(["GRID_ID", "label", "numer_dzie"])[
        ["GRID_ID", "label", "district_name"]].copy().reset_index(drop=True)
    lookup["display"] = lookup["label"] + " · " + lookup["district_name"]
    lookup["search"] = lookup["display"].map(normalize)
    return enriched, lookup


def normalize(text):
    text = unicodedata.normalize("NFKD", str(text).lower())
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"\b(ulica|ul)\.?\s+", "", text)
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def address_picker(key, catalog, label="Wyszukaj ulicę lub adres"):
    """Lokalne podpowiedzi, jak w wyszukiwarce : filtr + selectbox."""
    query = st.text_input(label, placeholder="np. Bronowicka 104", key=f"{key}_query")
    q = normalize(query)
    if len(q) < 2:
        st.caption("Wpisz co najmniej 2 znaki, aby zobaczyć podpowiedzi.")
        return None
    matches = catalog.loc[catalog["search"].str.contains(q, regex=False, na=False)].copy()
    if matches.empty:
        st.warning("Nie znaleziono adresu. Spróbuj wpisać fragment nazwy ulicy.")
        return None
    matches["first"] = matches["search"].str.startswith(q).astype(int)
    matches = matches.sort_values(["first", "display"], ascending=[False, True])
    matches = matches.drop_duplicates(["display", "GRID_ID"]).head(35)
    options = list(matches.index)
    selected = st.selectbox("Wybierz z podpowiedzi", options=options, key=f"{key}_pick",
                            format_func=lambda x: str(catalog.loc[x, "display"]))
    return str(catalog.loc[selected, "GRID_ID"])


def full_weights(partial):
    return {g: float(partial.get(g, 0)) for g in GROUPS}


def score_text(score):
    return f"{float(score):.1f}/100" if pd.notna(score) else "Brak oceny"


def place_title(row):
    d = row.get("district_name")
    district = str(d) if pd.notna(d) else "Kraków"
    street = row.get("street")
    if pd.notna(street) and str(street).strip():
        return f"ul. {street} · {district}"
    return f"{district} · okolica na mapie"


def evaluate_for_address(data, grid_id, radius, weights):
    from recommendation_engine import radius_grid_ids
    ids = radius_grid_ids(data, grid_id, radius)
    return evaluate_grid_area(data, ids, weights)


def result_chart_values(criteria):
    return pd.DataFrame([
        {"Kryterium": LABELS[g], "Ocena": round(100 * float(criteria[g]))}
        for g in USER_GROUPS if pd.notna(criteria.get(g, np.nan))
    ])


def show_result(result, title, radius_label):
    score = result["score"]
    st.markdown(f'<div class="k-score"><span>{html.escape(title)}</span><div>'
                f'<b>{score_text(score)}</b></div><span>Indeks dopasowania do wybranego profilu</span></div>',
                unsafe_allow_html=True)
    st.caption(f"{result['cells']} heksagonów · {radius_label}")
    st.caption("Wynik to umowny indeks 0–100, a nie procent szans na zadowolenie.")
    if pd.isna(score):
        st.warning("Za mało danych, aby wiarygodnie ocenić ten obszar.")
    detail = result_chart_values(result["criteria"])
    if not detail.empty:
        st.dataframe(detail, hide_index=True, use_container_width=True,
                     column_config={"Ocena": st.column_config.ProgressColumn(
                         "Ocena (0–100)", min_value=0, max_value=100, format="%d")})


def draw_top_map(top):
    center = [float(top["lat"].mean()), float(top["lon"].mean())]
    m = folium.Map(location=center, zoom_start=12, tiles="OpenStreetMap", control_scale=True)
    bounds = []
    for i, row in top.reset_index(drop=True).iterrows():
        lat, lon = float(row["lat"]), float(row["lon"])
        bounds.append([lat, lon])
        title = html.escape(place_title(row))
        folium.Circle([lat, lon], radius=500, color="#16877c", fill=True,
                      fill_opacity=.06, weight=1).add_to(m)
        folium.Marker([lat, lon], tooltip=f"{i+1}. {title}", popup=folium.Popup(
            f"<strong>{title}</strong><div>Ocena: {score_text(row['area_score'])}</div>", max_width=280),
            icon=folium.DivIcon(icon_size=(36, 36), icon_anchor=(18, 18), html=(
                '<div style="width:36px;height:36px;border:3px solid white;border-radius:50%;'
                'background:#0F756B;color:white;display:flex;align-items:center;justify-content:center;'
                f'box-shadow:0 3px 8px #2346;font-weight:750">{i+1}</div>'))).add_to(m)
    if len(bounds) > 1:
        m.fit_bounds(bounds, padding=(50, 50), max_zoom=15)
    components.html(m.get_root().render(), height=545, scrolling=False)


def render_location_panel():
    """Renderuje panel na istniejącej stronie po przetworzeniu jego mapy."""
    st.markdown("""
    <style>
    .block-container{max-width:1330px;padding-top:1.25rem;padding-bottom:3.8rem}
    [data-testid="stHeader"]{background:transparent}
    .k-header{background:linear-gradient(115deg,#173b49,#106e69);border-radius:22px;
      padding:30px 36px;color:white;margin:0 0 24px;box-shadow:0 12px 34px #103c4618}
    .k-header h1{color:white!important;font-size:2.25rem;margin:0 0 9px;letter-spacing:-.04em}
    .k-header p{color:#dcf3ee;font-size:1.02rem;margin:0;max-width:800px}
    .k-eyebrow{color:#acd9d2;font-size:.76rem;text-transform:uppercase;letter-spacing:.14em;font-weight:750;margin-bottom:8px}
    .k-section{font-size:1.35rem;font-weight:750;margin:19px 0 5px}
    .k-muted{opacity:.7;font-size:.9rem;margin:0 0 14px}
    .k-score{border-radius:18px;background:linear-gradient(125deg,#0f756b,#125957);
      color:white;padding:20px 23px;margin-bottom:14px}
    .k-score b{font-size:2.6rem;letter-spacing:-.04em;line-height:1.2}
    .k-score span{color:#d5f0eb;font-size:.9rem}
    .k-place{font-size:1.02rem;font-weight:750;margin-bottom:3px}
    .k-fine{font-size:.79rem;opacity:.75}
    div[data-testid="stForm"],div[data-testid="stVerticalBlockBorderWrapper"]{border-radius:16px}
    div[data-testid="stButton"] button, div[data-testid="stFormSubmitButton"] button{border-radius:11px;min-height:43px}
    [data-testid="stSlider"]{padding-bottom:5px}
    @media(max-width:760px){.k-header{padding:22px}.k-header h1{font-size:1.7rem}}
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="k-header">
      <div class="k-eyebrow">Kraków · Moje 15 minut</div>
      <h1>Odkryj swoje miejsce w Krakowie</h1>
      <p>Oceń zaznaczony na mapie obszar, a potem odkryj okolice, które najlepiej odpowiadają Twoim potrzebom.</p>
    </div>
    """, unsafe_allow_html=True)

    try:
        features, addresses = city_data()
    except (FileNotFoundError, ValueError, ImportError, OSError) as exc:
        st.error("Nie udało się wczytać bazy. Sprawdź pliki Parquet/CSV w katalogu projektu.")
        st.exception(exc)
        st.stop()

    # ────────────────────────────────────────────────────────── 1. OCENA MIEJSCA
    st.markdown('<div class="k-section">01 · Oceń wybraną lokalizację</div>', unsafe_allow_html=True)
    st.markdown('<p class="k-muted">Możesz przeanalizować obszar zaznaczony na mapie.</p>', unsafe_allow_html=True)

    with st.container(border=True):
        profile = st.selectbox("Dla kogo szukasz?", list(PROFILE_WEIGHTS),
                               index=1, key="analysis_profile")
        
        if profile == "Własne preferencje":
            custom = {}
            with st.expander("Ustaw własne priorytety", expanded=True):
                c1, c2 = st.columns(2)
                for j, group in enumerate(USER_GROUPS):
                    with (c1 if j % 2 == 0 else c2):
                        custom[group] = st.slider(LABELS[group], 0, 5, int(DEFAULT_WEIGHTS[group]),
                                                  key=f"analysis_weight_{group}")
            analysis_weights = full_weights(custom)
        else:
            analysis_weights = full_weights(PROFILE_WEIGHTS[profile])

        raw_selection = st.session_state.get("obszar_analityczny")
        try:
            marked_ids = hex_ids_from_selection(raw_selection)
        except ValueError as exc:
            st.warning(str(exc))
            marked_ids = []
            
        available_marked = int(features["GRID_ID"].isin(marked_ids).sum())
        if available_marked:
            st.success(f"Znaleziono {available_marked} heksagonów zaznaczonych na mapie.")
        else:
            st.info("Brak zaznaczonego obszaru. Wróć do „Analiza mapy” i narysuj obszar za pomocą narzędzia prostokąta lub poligonu.")

        evaluate_clicked = st.button("Oceń lokalizację", type="primary", use_container_width=True)

    if evaluate_clicked:
        if not any(analysis_weights.values()):
            st.warning("Wybierz przynajmniej jedno ważne kryterium.")
        elif not available_marked:
            st.warning("Najpierw zaznacz obszar na mapie.")
        else:
            st.session_state["analysis_request"] = {
                "grid_ids": marked_ids,
                "weights": analysis_weights, "profile": profile,
            }

    request = st.session_state.get("analysis_request")
    if request:
        same_target = set(request.get("grid_ids") or []) == set(marked_ids)
        if same_target:
            if any(analysis_weights.values()):
                request = {**request, "weights": analysis_weights, "profile": profile}
                st.session_state["analysis_request"] = request
        else:
            request = None
            st.info("Zmieniłeś zaznaczenie na mapie. Kliknij «Oceń lokalizację», aby je przeanalizować.")

    if request:
        try:
            ids = request["grid_ids"]
            radius_label = "zaznaczony obszar na mapie"
            area_result = evaluate_grid_area(features, ids, request["weights"], profile=request["profile"])
            show_result(area_result, f"Ocena dla: {request['profile']}", radius_label)
            st.caption("Odległości są geometryczne, liczone między centroidami heksagonów. Nie są to izochrony ani realne trasy piesze.")
        except (ValueError, IndexError) as exc:
            st.warning(str(exc))

    st.divider()

    # ────────────────────────────────────────────────────────── 2. TOP 5 (Optymalizacja: przeliczenie tylko po kliknięciu)
    st.markdown('<div class="k-section">02 · Odkryj najlepsze okolice</div>', unsafe_allow_html=True)
    st.markdown('<p class="k-muted">Wybierz miejsca w Krakowie, które chciałbyś rozważyć, i ustaw swoje priorytety.</p>', unsafe_allow_html=True)

    scope = st.radio("Gdzie szukamy?", ["Cały Kraków", "Wybrane dzielnice"],
                     horizontal=True, key="scope_choice")
    
    def apply_recommendation_profile():
        chosen = st.session_state.get("top5_profile")
        if chosen and chosen != "Własne preferencje":
            for group in USER_GROUPS:
                st.session_state[f"recommendation_{group}"] = int(PROFILE_WEIGHTS[chosen][group])

    st.selectbox("Szybki wybór preferencji (opcjonalnie)", list(PROFILE_WEIGHTS),
                 key="top5_profile", on_change=apply_recommendation_profile)

    with st.form("recommendation_form"):
        if scope == "Wybrane dzielnice":
            districts = st.multiselect("Wybierz jedną lub kilka dzielnic", list(DISTRICTS.values()),
                                       default=[], placeholder="Zacznij wpisywać, np. Krowodrza",
                                       help="Dzielnice nie są zaznaczone automatycznie. Możesz wybrać tylko jedną.")
        else:
            districts = list(DISTRICTS.values())
            st.caption("Przeszukamy wszystkie 18 dzielnic. Możesz też zawęzić wyniki do jednej lub kilku.")
        st.markdown("**Co jest dla Ciebie ważne?**")
        st.caption("0 — bez znaczenia · 5 — bardzo ważne")
        weights_form = {}
        left, right = st.columns(2, gap="large")
        for j, group in enumerate(USER_GROUPS):
            with (left if j % 2 == 0 else right):
                weights_form[group] = st.slider(LABELS[group], 0, 5, int(DEFAULT_WEIGHTS[group]),
                                                key=f"recommendation_{group}")
        recommend_clicked = st.form_submit_button("Pokaż 5 miejsc na mapie", type="primary",
                                                   use_container_width=True)

    if recommend_clicked:
        if scope == "Wybrane dzielnice" and not districts:
            st.warning("Wybierz przynajmniej jedną dzielnicę albo przełącz na „Cały Kraków”.")
        elif not any(weights_form.values()):
            st.warning("Ustaw przynajmniej jeden suwak powyżej zera.")
        else:
            with st.spinner("Przetwarzanie danych i wyszukiwanie najlepszych miejsc..."):
                recomm_weights = full_weights(weights_form)
                scored = score_city(features, recomm_weights)
                target_districts = districts if scope == "Wybrane dzielnice" else list(DISTRICTS.values())
                top_res = select_places_in_districts(scored, target_districts, count=5)
                st.session_state["cached_top"] = top_res
                st.session_state["cached_scope_desc"] = scope if scope == 'Cały Kraków' else ', '.join(target_districts)
                st.session_state["cached_weights"] = recomm_weights

    top = st.session_state.get("cached_top", pd.DataFrame())
    if not top.empty:
        st.caption(f"Wyniki dla: {st.session_state.get('cached_scope_desc', 'Kraków')}")
        draw_top_map(top)
        columns = st.columns(min(5, len(top)))
        for j, (_, row) in enumerate(top.reset_index(drop=True).iterrows()):
            with columns[j]:
                with st.container(border=True):
                    st.markdown(f"**{j+1}. {html.escape(str(row['district_name']))}**")
                    if pd.notna(row.get("street")):
                        st.caption("ul. " + str(row["street"]))
                    st.metric("Ocena", score_text(row["area_score"]))
        st.caption("Oceny są wskaźnikami modelu, nie procentem satysfakcji. Okręgi na mapie przedstawiają okolice 500 m.")
    else:
        st.info("Skonfiguruj preferencje powyżej i kliknij „Pokaż 5 miejsc na mapie”, aby wygenerować rekomendacje.")

    st.divider()

    # ────────────────────────────────────────────────────────── 3. PORÓWNANIE (Optymalizacja: obliczenia wyłącznie po kliknięciu przycisku)
    st.markdown('<div class="k-section">03 · Porównaj dwie lokalizacje</div>', unsafe_allow_html=True)
    st.markdown('<p class="k-muted">Zobacz, które z dwóch miejsc lepiej odpowiada Twoim priorytetom.</p>', unsafe_allow_html=True)

    with st.form("comparison_form"):
        comparison_mode = st.radio("Co porównujemy?", ["Propozycje z mapy", "Dwa własne adresy"],
                                   horizontal=True)
        compare_a = compare_b = None
        compare_labels = ("", "")
        compare_radius = 500

        if comparison_mode == "Propozycje z mapy":
            if len(top) >= 2:
                l, r = st.columns(2)
                fmt = lambda n: f"{n+1}. {place_title(top.iloc[n])}"
                with l:
                    a = st.selectbox("Pierwsze miejsce", list(range(len(top))), index=0, format_func=fmt, key="comp_a_idx")
                with r:
                    b = st.selectbox("Drugie miejsce", list(range(len(top))), index=1, format_func=fmt, key="comp_b_idx")
                if a != b:
                    compare_a = str(top.iloc[a]["GRID_ID"])
                    compare_b = str(top.iloc[b]["GRID_ID"])
                    compare_labels = (fmt(a), fmt(b))
            else:
                st.info("Najpierw wygeneruj co najmniej dwie rekomendacje w sekcji powyżej (02).")
        else:
            l, r = st.columns(2, gap="large")
            with l:
                compare_a = address_picker("compare_a", addresses, "Pierwszy adres")
            with r:
                compare_b = address_picker("compare_b", addresses, "Drugi adres")
            compare_radius = st.slider("Promień porównania (m)", 200, 3000, 800, 100, key="comp_radius_slider")

        compare_clicked = st.form_submit_button("Porównaj lokalizacje", type="primary", use_container_width=True)

    if compare_clicked:
        if compare_a is not None and compare_b is not None:
            if compare_a == compare_b:
                st.warning("To ten sam heksagon. Wybierz dwie różne lokalizacje.")
                st.session_state.pop("cached_comparison", None)
            else:
                current_weights = st.session_state.get("cached_weights", full_weights(DEFAULT_WEIGHTS))
                left_result = evaluate_for_address(features, compare_a, compare_radius, current_weights)
                right_result = evaluate_for_address(features, compare_b, compare_radius, current_weights)
                
                # Pobierz etykiety dla własnych adresów jeśli tryb to własne adresy
                if comparison_mode == "Dwa własne adresy":
                    a_label = addresses.loc[addresses["GRID_ID"].eq(compare_a), "display"].iloc[0] if not addresses.loc[addresses["GRID_ID"].eq(compare_a)].empty else "Lokalizacja A"
                    b_label = addresses.loc[addresses["GRID_ID"].eq(compare_b), "display"].iloc[0] if not addresses.loc[addresses["GRID_ID"].eq(compare_b)].empty else "Lokalizacja B"
                    comp_labels_final = (a_label, b_label)
                else:
                    comp_labels_final = compare_labels

                st.session_state["cached_comparison"] = {
                    "left": left_result, "right": right_result, "labels": comp_labels_final, "weights": current_weights
                }
        else:
            st.warning("Uzupełnij obie lokalizacje do porównania.")

    # Wyświetlanie zapamiętanych wyników porównania
    comp_data = st.session_state.get("cached_comparison")
    if comp_data:
        left_result = comp_data["left"]
        right_result = comp_data["right"]
        comp_labels_final = comp_data["labels"]
        c_weights = comp_data["weights"]

        a_col, b_col = st.columns(2, gap="large")
        for col, result, title in [(a_col, left_result, comp_labels_final[0]),
                                   (b_col, right_result, comp_labels_final[1])]:
            with col:
                with st.container(border=True):
                    st.markdown(f"<div class='k-place'>{html.escape(str(title))}</div>", unsafe_allow_html=True)
                    st.metric("Ocena dopasowania", score_text(result["score"]))
        rows = []
        for group in USER_GROUPS:
            if c_weights.get(group, 0) <= 0:
                continue
            left_value = left_result["criteria"].get(group, np.nan)
            right_value = right_result["criteria"].get(group, np.nan)
            rows.append({"Kryterium": LABELS[group], comp_labels_final[0]:
                         round(100 * left_value) if pd.notna(left_value) else None,
                         comp_labels_final[1]: round(100 * right_value) if pd.notna(right_value) else None})
        if rows:
            st.dataframe(pd.DataFrame(rows).set_index("Kryterium"), use_container_width=True)
        st.caption("Porównanie wykorzystuje zapisaną konfigurację kryteriów oraz wybrany promień.")

    st.caption("Źródła: otwarte dane Krakowa oraz lokalna baza adresowa. Dzielnice pustych heksagonów są przybliżane z adresów; niepewne przypadki pozostają nierozstrzygnięte. Ceny, hałas i wentylacja nie wpływają na punktację.")