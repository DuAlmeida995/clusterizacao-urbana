import pandas as pd
import geopandas as gpd
import warnings
warnings.filterwarnings("ignore")

print("1. Carregando os dados de chuva do INPE...")
df_chuva = pd.read_csv('clima_inpe_2022_limpo.csv')

# CORREÇÃO GEOGRÁFICA CRÍTICA:
# O satélite usa longitudes de 0 a 360 graus. O mapa do IBGE usa GPS normal (negativo).
# Precisamos subtrair 360 para os pontos "caírem" no lugar certo em cima do mapa!
df_chuva['longitude'] = df_chuva['longitude'] - 360

print("2. Transformando a tabela em mapa (GeoDataFrame)...")
gdf_chuva = gpd.GeoDataFrame(
    df_chuva, 
    geometry=gpd.points_from_xy(df_chuva.longitude, df_chuva.latitude),
    crs="EPSG:4326" 
)

print("3. Puxando o mapa de SP direto da internet (GeoJSON)...")
url_mapa = "https://raw.githubusercontent.com/tbrugz/geodata-br/master/geojson/geojs-35-mun.json"
mapa_sp = gpd.read_file(url_mapa)

print("4. Cruzando os pixels do satélite com os municípios (Spatial Join)...")
# O comando sjoin sobrepõe os pontos de chuva em cima do mapa das cidades
chuva_cidades = gpd.sjoin(gdf_chuva, mapa_sp, how="inner", predicate="within")

print("5. Consolidando a chuva média diária por cidade...")

chuva_diaria = chuva_cidades.groupby(['id', 'name', 'data'])['precipitacao_mm'].mean().reset_index()

print("6. Calculando o índice hídrico (Maior sequência de dias secos)...")
# Ordena por ID e data para garantir a cronologia correta
chuva_diaria = chuva_diaria.sort_values(['id', 'data'])

# Marca 1 se o dia foi seco, 0 se choveu
chuva_diaria['dia_seco'] = (chuva_diaria['precipitacao_mm'] < 1).astype(int)

# Identifica e agrupa os blocos de dias consecutivos sem chuva
chuva_diaria['quebra_seca'] = (chuva_diaria['dia_seco'] == 0).cumsum()

# Agrupa mantendo o 'id'
secas = chuva_diaria[chuva_diaria['dia_seco'] == 1].groupby(['id', 'name', 'quebra_seca']).size().reset_index(name='dias_consecutivos')

# Extrai o recorde (valor máximo de dias de estiagem) para cada município
df_final_kmeans = secas.groupby(['id', 'name'])['dias_consecutivos'].max().reset_index()

print("7. Salvando o arquivo final para o modelo!")
df_final_kmeans.to_csv('dias_consecutivos_seca.csv', index=False)
print("Sucesso! 'dias_consecutivos_seca.csv' gerado.")