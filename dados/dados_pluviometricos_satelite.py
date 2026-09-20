import xarray as xr
import pandas as pd
import pystac_client
import requests
from pathlib import Path
import warnings
import os

warnings.filterwarnings("ignore")
os.environ['PROJ_LIB'] = "/opt/conda/envs/geospatial/share/proj" # Recomendação do INPE para evitar alertas do Cartopy

def download_and_read_merge_stac(stac_service, start_date, end_date, output_dir="merge_data"):
    """
    Busca, baixa e lê dados diários do MERGE no catálogo STAC do INPE.
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    downloaded_files = []

    print(f"Buscando dados no INPE de {start_date} a {end_date}...")
    item_search = stac_service.search(
        datetime=f"{start_date}/{end_date}",
        collections=['prec_merge_daily-1']
    )

    for asset in item_search.items():
        if asset.assets['merge_daily'].href.endswith(".grib2"):
            file_url = asset.assets['merge_daily'].href
            filename = Path(file_url).name
            file_path = output_path / filename

            # Baixa o arquivo apenas se ele não existir na pasta
            if not file_path.exists():
                print(f"Baixando: {filename}")
                response = requests.get(file_url, stream=True)
                response.raise_for_status()
                with open(file_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk: f.write(chunk)
            
            downloaded_files.append(file_path)

    if not downloaded_files:
        raise FileNotFoundError("Nenhum arquivo encontrado no período especificado.")

    print(f"Lendo {len(downloaded_files)} arquivos com Xarray...")
    ds = xr.open_mfdataset(
        downloaded_files,
        engine='cfgrib',
        combine='nested',
        concat_dim='time',
        decode_timedelta=False
    )
    
    return ds

if __name__ == "__main__":
    # 1. Configura o serviço STAC do INPE
    service = pystac_client.Client.open('https://data.inpe.br/bdc/stac/v1/')
    
    # 2. Executa o download para o ano do projeto (2022)
    # Os arquivos brutos serão salvos em uma subpasta chamada 'dados_brutos_inpe'
    ds_satelite = download_and_read_merge_stac(service, '2022-01-01', '2022-12-31', 'dados_brutos_inpe')
    ds_satelite = ds_satelite.sortby('time')
    
# 3. Recorta o mapa ANTES de converter para Pandas para não explodir a RAM
    print("Recortando o cubo espacial apenas para São Paulo...")
    
    lat_min, lat_max = -26.0, -19.0
    lon_min, lon_max = -53.5, -44.0
    
    # Converte as longitudes negativas do Brasil para o padrão 0-360 do INPE
    lon_min_inpe = 360 + lon_min
    lon_max_inpe = 360 + lon_max
    
    # Ordena e corta
    if 'latitude' in ds_satelite.dims:
        ds_satelite = ds_satelite.sortby(['latitude', 'longitude'])
        ds_sp = ds_satelite.sel(latitude=slice(lat_min, lat_max), longitude=slice(lon_min_inpe, lon_max_inpe))
    else:
        ds_satelite = ds_satelite.sortby(['lat', 'lon'])
        ds_sp = ds_satelite.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min_inpe, lon_max_inpe))

    # 4. Converte a matriz MENOR para um DataFrame tabular
    print("Convertendo a matriz de SP para DataFrame Pandas...")
    df_chuva_bruto = ds_sp.to_dataframe().reset_index()
    
    # 5. Filtra o ano e limpa as colunas inúteis ANTES de salvar
    df_chuva_bruto = df_chuva_bruto[df_chuva_bruto['time'].dt.year == 2022]
    
    # Fica só com o que importa e renomeia
    df_chuva_limpo = df_chuva_bruto[['time', 'latitude', 'longitude', 'rdp']].copy()
    df_chuva_limpo.rename(columns={'rdp': 'precipitacao_mm', 'time': 'data'}, inplace=True)
    
    # 6. Salva o arquivo CSV final (agora muito mais leve!)
    nome_csv = 'clima_inpe_2022_limpo.csv'
    df_chuva_limpo.to_csv(nome_csv, index=False)
    
    print(f"Extração concluída! Tabela salva como '{nome_csv}'.")