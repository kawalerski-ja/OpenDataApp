import pandas as pd

# Wczytanie obecnego pliku CSV
addr_df = pd.read_csv('adresy.csv', sep=';')

# Opcjonalna optymalizacja typów danych (np. GRID_ID jako string, numery jako int)
addr_df['GRID_ID'] = addr_df['GRID_ID'].astype(str).str.strip()

# Zapis do skompresowanego pliku Parquet (domyślna kompresja Snappy)
addr_df.to_parquet('adresy.parquet', index=False)