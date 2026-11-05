import streamlit as st
import geopandas as gpd
import pandas as pd
import folium
import branca.colormap as cm
from streamlit_folium import st_folium
from folium.plugins import Draw
from shapely.geometry import shape
from location_panel import render_location_panel

st.set_page_config(page_title="Moje 15 minut", layout="wide")

st.title("Moje 15 minut: System Analiz Przestrzennych Krakowa")
st.caption("Analiza dostępności przestrzennej w oparciu o siatkę heksagonalną i buforowanie liniowo-przestrzenne.")

# Ładowanie danych
@st.cache_data
def load_data():
    gdf = gpd.read_parquet('krakow_master_grid_komplet.parquet')
    
    infra_flags = [c for c in gdf.columns if 'czy_zawiera_' in c or c == 'czy_przystanek_do_100m']
    if infra_flags:
        gdf['indeks_infrastruktury'] = gdf[[c for c in infra_flags if c in gdf.columns]].sum(axis=1)
    else:
        gdf['indeks_infrastruktury'] = 0

    name_cols = [c for c in gdf.columns if 'nazwa' in c or 'nazwy' in c]
    
    binary_name_mapping = {
        'pitnik': 'Pitnik',
        'toalete': 'Toaleta',
        'toaleta': 'Toaleta',
        'stojak': 'Stojak rowerowy',
        'biletomat': 'Biletomat'
    }
    
    def get_clean_tooltip_names(row):
        collected = []
        
        # Nazwane obiekty
        for col in name_cols:
            val = row.get(col)
            if pd.notna(val):
                text_val = str(val).strip()
                if text_val and text_val not in ['Brak', 'Brak obiektów', 'Brak informacji o przystankach']:
                    if ':' in text_val and not text_val.startswith('http'):
                        text_val = text_val.split(':', 1)[1].strip()
                    text_val = text_val.replace("(Brak statusu)", "").strip()
                    
                    is_green_col = 'korytarzy_zieleni' in col.lower()
                    
                    if text_val and text_val.lower() not in ['brak', 'nan', 'brak nazwy szlaku']:
                        if is_green_col:
                            collected.append(f"Korytarz zielony: {text_val}")
                        else:
                            collected.append(text_val)
                    elif is_green_col:
                        collected.append("Korytarz zielony")
                        
        if collected:
            unique_items = sorted(list(set(collected)))
            first_item = unique_items[0]
            
            # Limit 30 znaków
            if len(first_item) > 30:
                first_item = first_item[:27] + "..."
                
            if len(unique_items) > 1:
                return f"{first_item} (i {len(unique_items) - 1} inne)"
            else:
                return first_item
        
        # Jeśli brak nazw, sprawdzamy aktywne flagi binarne (0/1)
        active_binary = []
        for flag_col in infra_flags:
            if row.get(flag_col, 0) == 1:
                raw_key = flag_col.replace('czy_zawiera_', '').replace('czy_', '').lower()
                clean_name = binary_name_mapping.get(raw_key, raw_key.replace('_', ' ').capitalize())
                active_binary.append(clean_name)
                
        if active_binary:
            unique_binary = sorted(list(set(active_binary)))
            first_bin = unique_binary[0]
            
            if len(first_bin) > 30:
                first_bin = first_bin[:27] + "..."
                
            if len(unique_binary) > 1:
                return f"{first_bin} (i {len(unique_binary) - 1} inne)"
            else:
                return first_bin
            
        return "Brak"
    
    gdf['zbiorcze_nazwy_infra'] = gdf.apply(get_clean_tooltip_names, axis=1)

    if 'nazwy_przystankow' not in gdf.columns:
        p_cols = [c for c in gdf.columns if 'przystank' in c and c != 'czy_przystanek_do_100m']
        if p_cols:
            gdf['nazwy_przystankow'] = gdf[p_cols[0]]
        else:
            gdf['nazwy_przystankow'] = "Brak informacji o przystankach"

    gdf_4326 = gdf.to_crs(epsg=4326) if gdf.crs != "EPSG:4326" else gdf
    gdf_2180 = gdf.to_crs(epsg=2180)
    return gdf_4326, gdf_2180

@st.cache_data
def load_addresses():
    addr_df = pd.read_parquet('adresy.parquet')
    addr_df['label'] = addr_df['nazwa_ulic'] + ' ' + addr_df['numer_adre'].astype(str)
    return addr_df

with st.spinner("Ładowanie bazy danych przestrzennych..."):
    gdf_4326, gdf_2180 = load_data()
    addresses_df = load_addresses()

st.sidebar.header("Konfiguracja Widoku")

# Branding
st.sidebar.markdown("Projekt **Moje 15 minut**")
st.sidebar.markdown("Zespołu *Każdy ma swoje 15 minut*")
st.sidebar.markdown("---")

map_center = [50.0614, 19.9366]
map_zoom = 12
display_gdf = gdf_4326
selected_address = None

with st.sidebar.form(key="search_form"):
    st.markdown("### Wyszukiwarka Adresów")
    
    search_query = st.text_input("Filtruj nazwę ulicy:", "").strip().lower()
    
    if search_query:
        filtered_addresses = addresses_df[addresses_df['label'].str.lower().str.contains(search_query, na=False)]
    else:
        filtered_addresses = addresses_df
        
    if not filtered_addresses.empty:
        address_options = filtered_addresses['label'].unique()
    else:
        address_options = addresses_df['label'].unique()
        st.warning("Brak ulic pasujących do frazy. Pokazano pełną listę.")
        
    selected_address = st.selectbox(
        "Wybierz adres:",
        options=address_options
    )
        
    radius_m = st.slider(
        "Promień analizy (metry):",
        min_value=200, max_value=3000, value=800, step=100
    )
    
    submit_button = st.form_submit_button(label="Pokaż na mapie")

st.sidebar.markdown("### Warstwa Wizualna Mapy")
layers_dict = {
    'Infrastruktura (Zbiorcza)': 'indeks_infrastruktury',
    'Przystanki KMK': 'czy_przystanek_do_100m',
    'Prędkość wiatru (v4m)': 'wentyl_predkosc_v4m',
    'Potencjał wentylacji': 'wentyl_potencjal',
    'Hałas Drogowy (dB)': 'max_halas_dB',
    'Hałas Szynowy (dB)': 'max_halas_szyny_dB',
    'Dostępność dla niepełnosprawnych': 'score_niepelnosprawni'
}

selected_label = st.sidebar.selectbox(
    "Wybierz zmienną do wyrenderowania:",
    options=list(layers_dict.keys())
)
selected_column = layers_dict[selected_label]

st.sidebar.markdown("---")

show_all = st.sidebar.checkbox(
    "Wyświetl cały Kraków", 
    value=False,
    help="Uwaga: Wyświetlenie całego miasta mocno obciąża przeglądarkę i wydłuża czas renderowania mapy."
)
if not show_all and selected_address:
    target_grid_id = addresses_df[addresses_df['label'] == selected_address]['GRID_ID'].values[0]
    
    # Układ metryczny (EPSG:2180)
    target_row_2180 = gdf_2180[gdf_2180['GRID_ID'] == target_grid_id]
    
    if not target_row_2180.empty:
        target_geom = target_row_2180.geometry.iloc[0]
        
        centroid_2180 = target_geom.centroid
        point_gdf = gpd.GeoDataFrame(geometry=[centroid_2180], crs="EPSG:2180").to_crs("EPSG:4326")
        centroid_4326 = point_gdf.geometry.iloc[0]
        
        map_center = [centroid_4326.y, centroid_4326.x]
        map_zoom = 15
        
        # Buforowanie i przycięcie przestrzenne
        buffered_geom = target_geom.buffer(radius_m)
        possible_matches_index = list(gdf_2180.sindex.query(buffered_geom, predicate="intersects"))
        subset_2180 = gdf_2180.iloc[possible_matches_index]
        
        exact_mask = subset_2180.geometry.distance(target_geom) <= radius_m
        valid_grid_ids = subset_2180[exact_mask]['GRID_ID'].tolist()
        
        display_gdf = gdf_4326[gdf_4326['GRID_ID'].isin(valid_grid_ids)]
colormap = None
if selected_column not in ['indeks_infrastruktury', 'czy_przystanek_do_100m']:
    valid_vals = display_gdf[selected_column].dropna()
    if not valid_vals.empty and pd.api.types.is_numeric_dtype(valid_vals):
        vmin, vmax = float(valid_vals.min()), float(valid_vals.max())
        if vmin == vmax:
            vmax = vmin + 1.0
        
        if 'halas' in selected_column:
            colors = ['#2ca02c', '#ff7f0e', '#d62728']
        elif 'wentyl' in selected_column:
            colors = ['#d73027', '#fee08b', '#1a9850']
        else:
            colors = ['#fee8c8', '#e34a33']
            
        colormap = cm.LinearColormap(colors=colors, vmin=vmin, vmax=vmax)

m = folium.Map(location=map_center, zoom_start=map_zoom, tiles='OpenStreetMap')

def style_function(feature):
    val = feature['properties'].get(selected_column)
    
    if val is None or pd.isna(val) or val == 'Brak danych' or val == 0:
        return {'fillColor': '#000000', 'color': '#000000', 'weight': 0, 'fillOpacity': 0.0}
    
    if selected_column == 'indeks_infrastruktury':
        return {'fillColor': '#31a354', 'color': '#238b45', 'weight': 0.5, 'fillOpacity': 0.65}
    elif selected_column == 'czy_przystanek_do_100m':
        return {'fillColor': '#2b8cbe', 'color': '#045a8d', 'weight': 0.5, 'fillOpacity': 0.65}
    
    if colormap is not None and isinstance(val, (int, float)):
        color = colormap(val)
        return {'fillColor': color, 'color': color, 'weight': 0.5, 'fillOpacity': 0.7}
        
    return {'fillColor': '#756bb1', 'color': '#54278f', 'weight': 0.5, 'fillOpacity': 0.65}

Draw(
    draw_options={'polyline': False, 'marker': False, 'circle': False, 'circlemarker': False, 'rectangle': True, 'polygon': True},
    edit_options={'edit': False}
).add_to(m)

if selected_column == 'indeks_infrastruktury':
    tooltip_field = 'zbiorcze_nazwy_infra'
    tooltip_alias = 'Obiekty:'
elif selected_column == 'czy_przystanek_do_100m':
    tooltip_field = 'nazwy_przystankow'
    tooltip_alias = 'Przystanki:'
else:
    tooltip_field = selected_column
    tooltip_alias = f'{selected_label}:'

folium.GeoJson(
    display_gdf,
    name="Heksagony",
    style_function=style_function,
    tooltip=folium.GeoJsonTooltip(
        fields=[tooltip_field],
        aliases=[tooltip_alias],
        localize=True,
        max_width=300
    )
).add_to(m)

if colormap is not None:
    colormap.caption = f"Skala: {selected_label}"
    colormap.add_to(m)

col_map, col_stats = st.columns([2, 1])

with col_map:
    if show_all:
        st.info("Tryb pełnego widoku: Wczytano wszystkie heksagony dla całego Krakowa.")
    else:
        st.info(f"Mój obszar 15-minutowy: Wczytano heksagony w promieniu {radius_m} m od adresu: {selected_address}. Aby zmienić obszar, wpisz nowy adres w panelu bocznym.")
        
    st.subheader(f"Zaznacz obszar na mapie (Widocznych heksagonów: {len(display_gdf)})")
    map_data = st_folium(m, width="100%", height=600)

with col_stats:
    st.subheader("Panel Podsumowania Obszaru")
    
    drawn_geojson = None
    if map_data and 'last_active_drawing' in map_data and map_data['last_active_drawing']:
        drawn_geojson = map_data['last_active_drawing']
    
    if drawn_geojson:
        polygon_shape = shape(drawn_geojson['geometry'])
        selected_hexagons = display_gdf[display_gdf.geometry.intersects(polygon_shape)]
        count_hex = len(selected_hexagons)
        
        if count_hex > 0:
            st.success(f"Zaznaczono heksagonów: {count_hex}")
            
            st.session_state['obszar_analityczny'] = selected_hexagons.drop(columns='geometry', errors='ignore')
            
            # MPZP
            mpzp_cols = [c for c in selected_hexagons.columns if 'mpzp' in c.lower() or 'plan' in c.lower()]
            st.markdown("**MPZP / Uwarunkowania planistyczne:**")
            if mpzp_cols:
                col_name = mpzp_cols[0]
                raw_mpzp = selected_hexagons[col_name]
                valid_mpzp = raw_mpzp[
                    raw_mpzp.notna() & 
                    (raw_mpzp.astype(str).str.strip() != '') & 
                    (raw_mpzp.astype(str).str.lower() != 'brak') &
                    (raw_mpzp.astype(str).str.lower() != 'nan')
                ]
                unique_plans_count = valid_mpzp.nunique()
                if unique_plans_count > 0:
                    st.write(f"Liczba zidentyfikowanych typów planów w obszarze: {unique_plans_count}")
                    with st.expander("Zobacz pełną listę planów w zaznaczeniu"):
                        st.write(valid_mpzp.unique().tolist())
                else:
                    st.write("Brak obowiązującego planu MPZP w zaznaczonym obszarze.")
            else:
                st.write("Brak dedykowanej kolumny MPZP w zbiorze danych.")

            st.markdown("---")

            # Ceny nieruchomości
            st.markdown("**Rynek Nieruchomości:**")
            if 'cena_sr_za_m2' in selected_hexagons.columns:
                valid_prices = selected_hexagons['cena_sr_za_m2'].dropna()
                if not valid_prices.empty:
                    st.metric("Średnia cena m²", f"{valid_prices.mean():,.0f} PLN/m²".replace(",", " "))
                    if not valid_prices.min() == valid_prices.max():
                        st.write(f"Zakres cenowy: {valid_prices.min():,.0f} - {valid_prices.max():,.0f} PLN/m²".replace(",", " "))
                else:
                    st.metric("Średnia cena m²", "Brak danych")
            else:
                st.write("Brak danych cenowych.")

            st.markdown("---")

            # Warunki Środowiskowe
            st.markdown("**Warunki Środowiskowe (Wartości Średnie):**")
            if 'max_halas_dB' in selected_hexagons.columns:
                valid_noise = selected_hexagons['max_halas_dB'].dropna()
                st.write(f"- Hałas drogowy: {valid_noise.mean():.1f} dB" if not valid_noise.empty else "- Hałas drogowy: Brak pomiarów")
            if 'max_halas_szyny_dB' in selected_hexagons.columns:
                valid_rail = selected_hexagons['max_halas_szyny_dB'].dropna()
                st.write(f"- Hałas szynowy: {valid_rail.mean():.1f} dB" if not valid_rail.empty else "- Hałas szynowy: Brak pomiarów")
            if 'wentyl_predkosc_v4m' in selected_hexagons.columns:
                valid_wind = selected_hexagons['wentyl_predkosc_v4m'].dropna()
                st.write(f"- Prędkość wiatru (v4m): {valid_wind.mean():.2f} m/s" if not valid_wind.empty else "- Prędkość wiatru: Brak danych")
            if 'wentyl_potencjal' in selected_hexagons.columns:
                valid_pot = selected_hexagons['wentyl_potencjal'].dropna()
                st.write(f"- Potencjał wentylacji: {valid_pot.mean():.2f}" if not valid_pot.empty else "- Potencjał wentylacji: Brak danych")

            st.markdown("---")

            # Zasoby Infrastrukturalne
            st.markdown("**Zasoby Infrastrukturalne w Obszarze:**")
            
            category_mapping = {
                'sport': 'Sport i Rekreacja',
                'edukacja': 'Edukacja',
                'zdrowie': 'Zdrowie i Medycyna',
                'mops': 'Opieka Społeczna (MOPS)',
                'korytarzy_zieleni': 'Korytarze i Tereny Zielone',
                'cas': 'Centra Aktywności Seniora (CAS)',
                'pitnik': 'Pitniki',
                'przystank': 'Przystanki Komunikacji Miejskiej'
            }

            name_cols = [
                c for c in selected_hexagons.columns 
                if ('nazwa' in c or 'nazwy' in c) and c != 'zbiorcze_nazwy_infra'
            ]
            
            raw_categorized = {}
            global_seen = set()

            def extract_clean_names(raw_value, is_green_corridor=False):
                text_val = str(raw_value).strip()
                if not text_val or text_val in ['Brak', 'Brak obiektów', 'Brak informacji o przystankach']:
                    return []
                
                if ':' in text_val and not text_val.startswith('http'):
                    text_val = text_val.split(':', 1)[1].strip()
                
                text_val = text_val.replace("(Brak statusu)", "").strip()
                
                if text_val.lower() in ['brak nazwy szlaku', 'brak', 'nan', '']:
                    if is_green_corridor:
                        return ["Korytarz zielony"]
                    return []
                
                split_parts = [p.strip() for p in text_val.replace('|', ',').split(',') if p.strip()]
                valid_parts = []
                for p in split_parts:
                    p_lower = p.lower()
                    if p_lower in ['brak nazwy szlaku', 'brak', 'nan', '']:
                        if is_green_corridor:
                            valid_parts.append("Korytarz zielony")
                    elif p:
                        valid_parts.append(p)
                        
                return valid_parts if valid_parts else (["Korytarz zielony"] if is_green_corridor else [])

            # Kategoryzacja głównych zasobów
            for col in name_cols:
                matched_cat = None
                for key, label in category_mapping.items():
                    if key in col.lower():
                        matched_cat = label
                        break
                
                if matched_cat is None:
                    continue
                
                if matched_cat not in raw_categorized:
                    raw_categorized[matched_cat] = set()

                is_green = ('korytarzy_zieleni' in col.lower())

                for val in selected_hexagons[col].dropna():
                    items = extract_clean_names(val, is_green_corridor=is_green)
                    for item in items:
                        raw_categorized[matched_cat].add(item)
                        global_seen.add(item.lower())

            other_items = set()
            for col in name_cols:
                is_main = any(key in col.lower() for key in category_mapping.keys())
                if not is_main:
                    for val in selected_hexagons[col].dropna():
                        items = extract_clean_names(val)
                        for item in items:
                            item_lower = item.lower()
                            if len(item) >= 4 and item_lower not in global_seen:
                                other_items.add(item)
                                global_seen.add(item_lower)

            if other_items:
                raw_categorized['Inne obiekty'] = other_items

            total_found_objects = 0
            for cat, items in raw_categorized.items():
                if items:
                    sorted_items = sorted(list(items))
                    total_found_objects += len(sorted_items)
                    with st.expander(f"{cat} ({len(sorted_items)})"):
                        for item in sorted_items:
                            st.markdown(f"- {item}")

            if total_found_objects == 0:
                st.write("Brak zidentyfikowanych obiektów w zaznaczonym obszarze.")

        else:
            st.warning("Narysowany obszar nie pokrywa żadnego widocznego heksagonu.")
            if 'obszar_analityczny' in st.session_state:
                del st.session_state['obszar_analityczny']
    else:
        st.info("Użyj narzędzia rysowania na mapie (kwadrat lub poligon w lewym górnym rogu mapy), aby wygenerować podsumowanie obszaru.")

# Panel analizy lokalizacji i rekomendacji
st.divider()

render_location_panel()