import streamlit as st
import geopandas as gpd
import pandas as pd
import folium
import branca.colormap as cm
from streamlit_folium import st_folium
from folium.plugins import Draw
from shapely.geometry import shape

st.set_page_config(page_title="Krakow Urban DSS - Analiza Heksagonalna", layout="wide")

st.title("System Analiz Przestrzennych Krakowa (Siatka Heksagonalna)")

@st.cache_data
def load_data():
    gdf = gpd.read_parquet('krakow_master_grid_komplet.parquet')
    
    infra_flags = [c for c in ['czy_zawiera_sport', 'czy_zawiera_edukacje', 'czy_zawiera_mops', 'czy_zawiera_pitnik'] if c in gdf.columns]
    if infra_flags:
        gdf['indeks_infrastruktury'] = gdf[infra_flags].sum(axis=1)
    else:
        gdf['indeks_infrastruktury'] = 0

    name_cols = [c for c in ['nazwy_edukacja', 'nazwy_sport', 'nazwy_mops'] if c in gdf.columns]
    def get_combined_names(row):
        collected = []
        for col in name_cols:
            val = row.get(col)
            if pd.notna(val) and val != 'Brak':
                collected.append(str(val))
        return " | ".join(collected) if collected else "Brak nazwanych obiektow"
    
    gdf['zbiorcze_nazwy_infra'] = gdf.apply(get_combined_names, axis=1)

    # Zabezpieczenie nazwy kolumny z przystankami
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
    addr_df = pd.read_csv('adresy.csv', sep=';')
    addr_df['GRID_ID'] = addr_df['GRID_ID'].astype(str).str.strip()
    addr_df['label'] = addr_df['nazwa_ulic'] + ' ' + addr_df['numer_adre'].astype(str)
    return addr_df

with st.spinner("Ladowanie bazy danych przestrzennych..."):
    gdf_4326, gdf_2180 = load_data()
    addresses_df = load_addresses()

st.sidebar.header("Konfiguracja Widoku")

show_all = st.sidebar.checkbox("Wyswietl caly Krakow", value=False)

map_center = [50.0614, 19.9366]
map_zoom = 12
display_gdf = gdf_4326
selected_address = None

with st.sidebar.form(key="search_form"):
    st.markdown("### Wyszukiwarka Adresow")
    search_query = st.text_input("Filtruj nazwe ulicy:", "").strip().lower()
    
    if search_query:
        filtered_addresses = addresses_df[addresses_df['label'].str.lower().str.contains(search_query, na=False)]
    else:
        filtered_addresses = addresses_df
        
    if not filtered_addresses.empty:
        selected_address = st.selectbox(
            "Wybierz adres:",
            options=filtered_addresses['label'].unique()
        )
    else:
        st.warning("Brak ulic pasujacych do frazy.")
        
    radius_m = st.slider(
        "Promien analizy (metry):",
        min_value=200, max_value=3000, value=800, step=100
    )
    
    submit_button = st.form_submit_button(label="Pokaz na mapie")

if not show_all and selected_address:
    target_grid_id = addresses_df[addresses_df['label'] == selected_address]['GRID_ID'].values[0]
    target_row = gdf_2180[gdf_2180['GRID_ID'] == target_grid_id]
    
    if not target_row.empty:
        target_geom = target_row.geometry.iloc[0]
        centroid_4326 = gdf_4326[gdf_4326['GRID_ID'] == target_grid_id].geometry.centroid.iloc[0]
        map_center = [centroid_4326.y, centroid_4326.x]
        map_zoom = 15
        
        buffered_geom = target_geom.buffer(radius_m)
        possible_matches_index = list(gdf_2180.sindex.query(buffered_geom, predicate="intersects"))
        subset_2180 = gdf_2180.iloc[possible_matches_index]
        
        exact_mask = subset_2180.geometry.distance(target_geom) <= radius_m
        valid_grid_ids = subset_2180[exact_mask]['GRID_ID'].tolist()
        
        display_gdf = gdf_4326[gdf_4326['GRID_ID'].isin(valid_grid_ids)]

st.sidebar.markdown("### Warstwa Wizualna Mapy")
layers_dict = {
    'Infrastruktura (Szkoly, Sport, MOPS...)': 'indeks_infrastruktury',
    'Przystanki KMK': 'czy_przystanek_do_100m',
    'Predkosc wiatru (v4m)': 'wentyl_predkosc_v4m',
    'Potencjal wentylacji': 'wentyl_potencjal',
    'Halas Drogowy (dB)': 'max_halas_dB',
    'Halas Szynowy (dB)': 'max_halas_szyny_dB',
    'Dostepnosc dla niepelnosprawnych': 'score_niepelnosprawni'
}

selected_label = st.sidebar.selectbox(
    "Wybierz zmienna do wyrenderowania:",
    options=list(layers_dict.keys())
)
selected_column = layers_dict[selected_label]

# Przygotowanie skali kolorow (heatmapy) dla zmiennych ciaglych
colormap = None
if selected_column not in ['indeks_infrastruktury', 'czy_przystanek_do_100m']:
    valid_vals = display_gdf[selected_column].dropna()
    if not valid_vals.empty and pd.api.types.is_numeric_dtype(valid_vals):
        vmin, vmax = float(valid_vals.min()), float(valid_vals.max())
        if vmin == vmax:
            vmax = vmin + 1.0
        
        if 'halas' in selected_column:
            colors = ['#2ca02c', '#ff7f0e', '#d62728'] # Zielony -> Zolty -> Czerwony
        elif 'wentyl' in selected_column:
            colors = ['#d73027', '#fee08b', '#1a9850'] # Czerwony -> Zolty -> Zielony
        else:
            colors = ['#fee8c8', '#e34a33'] # Jasny -> Ciemny
            
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
    
    # Renderowanie efektu heatmapy dla zmiennych ciaglych
    if colormap is not None and isinstance(val, (int, float)):
        color = colormap(val)
        return {'fillColor': color, 'color': color, 'weight': 0.5, 'fillOpacity': 0.7}
        
    return {'fillColor': '#756bb1', 'color': '#54278f', 'weight': 0.5, 'fillOpacity': 0.65}

Draw(
    draw_options={'polyline': False, 'marker': False, 'circle': False, 'circlemarker': False, 'rectangle': True, 'polygon': True},
    edit_options={'edit': False}
).add_to(m)

# Ustawienie odpowiedniego pola w dymku (bez GRID_ID)
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
        localize=True
    )
).add_to(m)

# Dodanie legendy heatmapy do mapy, jesli dotyczy zmiennej ciaglej
if colormap is not None:
    colormap.caption = f"Skala: {selected_label}"
    colormap.add_to(m)

col_map, col_stats = st.columns([2, 1])

with col_map:
    st.subheader(f"Mapa Interaktywna (Widocznych heksagonow: {len(display_gdf)})")
    map_data = st_folium(m, width="100%", height=600)

with col_stats:
    st.subheader("Statystyki Zaznaczonego Obszaru")
    
    drawn_geojson = None
    if map_data and 'last_active_drawing' in map_data and map_data['last_active_drawing']:
        drawn_geojson = map_data['last_active_drawing']
    
    if drawn_geojson:
        polygon_shape = shape(drawn_geojson['geometry'])
        selected_hexagons = display_gdf[display_gdf.geometry.intersects(polygon_shape)]
        count_hex = len(selected_hexagons)
        
        if count_hex > 0:
            st.success(f"Zaznaczono heksagonow: {count_hex}")
            
            if 'cena_sr_za_m2' in selected_hexagons.columns:
                valid_prices = selected_hexagons['cena_sr_za_m2'].dropna()
                if not valid_prices.empty:
                    st.metric("Srednia cena m2", f"{valid_prices.mean():,.0f} PLN/m2".replace(",", " "))
                else:
                    st.metric("Srednia cena m2", "Brak danych")
                    
            if 'max_halas_dB' in selected_hexagons.columns:
                valid_noise = selected_hexagons['max_halas_dB'].dropna()
                if not valid_noise.empty:
                    st.metric("Sredni halas drogowy", f"{valid_noise.mean():.1f} dB")
                else:
                    st.metric("Sredni halas drogowy", "Brak pomiarow")
                    
            if 'wentyl_predkosc_v4m' in selected_hexagons.columns:
                valid_wind = selected_hexagons['wentyl_predkosc_v4m'].dropna()
                if not valid_wind.empty:
                    st.metric("Srednia predkosc wiatru", f"{valid_wind.mean():.2f} m/s")
        else:
            st.warning("Narysowany obszar nie pokrywa zadnego widocznego heksagonu.")
    else:
        st.info("Wyszukaj adres lub uzyj narzedzia rysowania na mapie.")