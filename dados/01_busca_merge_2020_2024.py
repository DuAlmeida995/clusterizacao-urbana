"""
PASSO 1 - Busca os dados diarios do MERGE/INPE (2020 a 2024) e gera CSVs.

O que este script faz, em ordem:
  1. Baixa os arquivos .grib2 diarios do MERGE (catalogo STAC do INPE), um por dia.
  2. Recorta a grade para o estado de Sao Paulo.
  3. Faz o cruzamento espacial (pixel -> municipio) UMA vez e calcula a
     precipitacao media diaria de cada municipio (mesmo metodo do trabalho: media dos
     pixels dentro do poligono municipal).
  4. Calcula o CDD (maior sequencia de dias secos, dia seco = < 1 mm) de cada ano.

Arquivos gerados (na pasta dados/):
  - merge_chuva_diaria_municipios_2020_2024.csv  (id, name, data, precipitacao_mm, n_pixels)
  - cdd_anual_municipios_2020_2024.csv           (id, name, ano, cdd, dias_sem_dado)
  - centroides_municipios.csv                    (id, name, lat, lon) -> usado na imputacao do passo 2

Como rodar (no terminal, dentro da pasta dados/):
  python 01_busca_merge_2020_2024.py
  python 01_busca_merge_2020_2024.py --anos 2020 2021          # so alguns anos
  python 01_busca_merge_2020_2024.py --apagar-brutos           # apaga os .grib2 apos cada ano (economiza disco)

Bibliotecas: pandas numpy xarray cfgrib eccodes pystac-client geopandas requests
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PASTA = Path(__file__).resolve().parent
URL_STAC = "https://data.inpe.br/bdc/stac/v1/"
URL_MALHA_IBGE = ("https://servicodados.ibge.gov.br/api/v3/malhas/estados/35"
                  "?resolucao=5&intrarregiao=municipio&formato=application/vnd.geo+json")
URL_NOMES_IBGE = "https://servicodados.ibge.gov.br/api/v1/localidades/estados/35/municipios"

# Caixa que contem o estado de SP (mesma usada no seu script de 2022)
LAT_MIN, LAT_MAX = -26.0, -19.0
LON_MIN, LON_MAX = -53.5, -44.0
LIMIAR_DIA_SECO_MM = 1.0  # padrao ETCCDI e o mesmo usado no trabalho


# ----------------------------------------------------------------------------
# 1) Malha municipal
# ----------------------------------------------------------------------------
def carregar_malha():
    import geopandas as gpd
    import requests

    print("Baixando a malha municipal de SP (IBGE)...")
    r = requests.get(URL_MALHA_IBGE, timeout=120)
    r.raise_for_status()
    mapa = gpd.GeoDataFrame.from_features(r.json()["features"], crs="EPSG:4326")

    # a malha vem com o codigo em 'codarea'; os nomes vem de outra API
    col_id = "codarea" if "codarea" in mapa.columns else "id"
    mapa = mapa.rename(columns={col_id: "id"})
    mapa["id"] = mapa["id"].astype(str).str[:7].astype(int)
    try:
        rn = requests.get(URL_NOMES_IBGE, timeout=120)
        rn.raise_for_status()
        nomes = pd.DataFrame([{"id": m["id"], "name": m["nome"]} for m in rn.json()])
        mapa = mapa.merge(nomes, on="id", how="left")
    except Exception as e:  # nomes sao opcionais
        print("Aviso: nao consegui os nomes dos municipios:", e)
        mapa["name"] = mapa["id"].astype(str)
    return mapa[["id", "name", "geometry"]]


def salvar_centroides(mapa):
    # centroide em projecao metrica (evita erro de centroide em graus)
    c = mapa.to_crs("EPSG:5880").centroid.to_crs("EPSG:4326")
    df = pd.DataFrame({"id": mapa["id"], "name": mapa["name"], "lat": c.y, "lon": c.x})
    df.to_csv(PASTA / "centroides_municipios.csv", index=False)
    return df


# ----------------------------------------------------------------------------
# 2) Download dos arquivos diarios
# ----------------------------------------------------------------------------
def baixar_ano(ano, pasta_brutos):
    import pystac_client
    import requests

    pasta_brutos.mkdir(parents=True, exist_ok=True)
    servico = pystac_client.Client.open(URL_STAC)
    busca = servico.search(datetime=f"{ano}-01-01/{ano}-12-31", collections=["prec_merge_daily-1"])

    arquivos = []
    for item in busca.items():
        href = item.assets["merge_daily"].href
        if not href.endswith(".grib2"):
            continue
        destino = pasta_brutos / Path(href).name
        if not destino.exists() or destino.stat().st_size == 0:
            ok = False
            for tentativa in range(3):
                try:
                    with requests.get(href, stream=True, timeout=120) as r:
                        r.raise_for_status()
                        with open(destino, "wb") as f:
                            for pedaco in r.iter_content(chunk_size=1024 * 1024):
                                f.write(pedaco)
                    ok = True
                    break
                except Exception as e:
                    print(f"  falha em {destino.name} (tentativa {tentativa + 1}/3): {e}")
            if not ok:
                destino.unlink(missing_ok=True)
                continue
        arquivos.append(destino)
    if not arquivos:
        raise FileNotFoundError(f"Nenhum arquivo MERGE encontrado para {ano}.")
    return sorted(arquivos)


# ----------------------------------------------------------------------------
# 3) Pixel -> municipio (feito uma unica vez) e media diaria por municipio
# ----------------------------------------------------------------------------
def abrir_ano(arquivos):
    import xarray as xr

    ds = xr.open_mfdataset(arquivos, engine="cfgrib", combine="nested",
                           concat_dim="time", decode_timedelta=False)
    ds = ds.sortby("time")
    nome_lat = "latitude" if "latitude" in ds.dims else "lat"
    nome_lon = "longitude" if "longitude" in ds.dims else "lon"
    ds = ds.sortby([nome_lat, nome_lon])
    # o INPE usa longitude de 0 a 360
    ds = ds.sel({nome_lat: slice(LAT_MIN, LAT_MAX),
                 nome_lon: slice(360 + LON_MIN, 360 + LON_MAX)})
    return ds, nome_lat, nome_lon


def mapa_pixel_municipio(ds, nome_lat, nome_lon, mapa):
    """Retorna (indices dos pixels dentro de municipios, id do municipio de cada pixel)."""
    import geopandas as gpd

    lats = ds[nome_lat].values
    lons = ds[nome_lon].values - 360
    LON, LAT = np.meshgrid(lons, lats)
    pix = gpd.GeoDataFrame(
        {"pos": np.arange(LON.size)},
        geometry=gpd.points_from_xy(LON.ravel(), LAT.ravel()), crs="EPSG:4326")
    j = gpd.sjoin(pix, mapa[["id", "geometry"]], how="inner", predicate="within")
    return j["pos"].values, j["id"].values


def precip_diaria_municipios(ds, var, pos, ids_pix, nomes, ano):
    chuva = ds[var].values  # (tempo, lat, lon)
    chuva = chuva.reshape(chuva.shape[0], -1)[:, pos].astype("float64")
    chuva[chuva < 0] = np.nan  # valores negativos = sem dado
    tabela = pd.DataFrame(chuva.T)
    tabela["id"] = ids_pix
    media = tabela.groupby("id").mean(numeric_only=True)          # media dos pixels
    npix = tabela.groupby("id").size().rename("n_pixels")
    datas = pd.to_datetime(ds["time"].values).normalize()
    media.columns = datas
    # calendario completo do ano: dia que nao veio no download vira NaN
    # (nao e contado como dia seco e interrompe a sequencia)
    calendario = pd.date_range(f"{ano}-01-01", f"{ano}-12-31", freq="D")
    media = media.reindex(columns=calendario)
    longo = media.reset_index().melt(id_vars="id", var_name="data", value_name="precipitacao_mm")
    longo = longo.merge(npix, on="id").merge(nomes, on="id", how="left")
    longo = longo[pd.to_datetime(longo["data"]).dt.year == ano]
    return longo[["id", "name", "data", "precipitacao_mm", "n_pixels"]].sort_values(["id", "data"])


# ----------------------------------------------------------------------------
# 4) CDD
# ----------------------------------------------------------------------------
def maior_sequencia_seca(chuva):
    """Maior sequencia de dias com chuva < 1 mm. Dia sem dado (NaN) interrompe a sequencia."""
    seco = np.asarray(chuva) < LIMIAR_DIA_SECO_MM   # NaN -> False
    maior = atual = 0
    for s in seco:
        atual = atual + 1 if s else 0
        maior = max(maior, atual)
    return maior


def cdd_anual(longo):
    linhas = []
    for (i, n, ano), g in longo.assign(ano=pd.to_datetime(longo["data"]).dt.year).groupby(["id", "name", "ano"]):
        g = g.sort_values("data")
        linhas.append({"id": i, "name": n, "ano": ano,
                       "cdd": maior_sequencia_seca(g["precipitacao_mm"].values),
                       "dias_observados": int(g["precipitacao_mm"].notna().sum()),
                       "dias_sem_dado": int(g["precipitacao_mm"].isna().sum())})
    return pd.DataFrame(linhas)


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anos", nargs="+", type=int, default=[2020, 2021, 2022, 2023, 2024])
    ap.add_argument("--brutos", default=str(PASTA / "dados_brutos_inpe"))
    ap.add_argument("--apagar-brutos", action="store_true")
    a = ap.parse_args()

    mapa = carregar_malha()
    nomes = salvar_centroides(mapa)[["id", "name"]]
    print(f"Malha: {len(mapa)} municipios. Centroides salvos.")

    partes = []
    for ano in a.anos:
        arq_ano = PASTA / f"_parcial_v2_chuva_{ano}.csv"
        if arq_ano.exists():                      # permite retomar de onde parou
            print(f"{ano}: usando resultado parcial ja salvo.")
            partes.append(pd.read_csv(arq_ano))
            continue
        print(f"\n=== {ano} ===")
        arquivos = baixar_ano(ano, Path(a.brutos))
        print(f"{len(arquivos)} arquivos diarios. Lendo...")
        ds, nlat, nlon = abrir_ano(arquivos)
        var = "rdp" if "rdp" in ds.data_vars else list(ds.data_vars)[0]
        print(f"Variavel de precipitacao: {var}")
        pos, ids_pix = mapa_pixel_municipio(ds, nlat, nlon, mapa)
        print(f"{len(pos)} pixels dentro de {len(set(ids_pix))} municipios.")
        longo = precip_diaria_municipios(ds, var, pos, ids_pix, nomes, ano)
        longo.to_csv(arq_ano, index=False)
        partes.append(longo)
        ds.close()
        if a.apagar_brutos:
            for f in arquivos:
                f.unlink(missing_ok=True)

    todo = pd.concat(partes, ignore_index=True).sort_values(["id", "data"])
    todo["precipitacao_mm"] = todo["precipitacao_mm"].round(3)
    todo.to_csv(PASTA / "merge_chuva_diaria_municipios_2020_2024.csv", index=False)

    cdd = cdd_anual(todo)
    cdd.to_csv(PASTA / "cdd_anual_municipios_2020_2024.csv", index=False)

    faltantes = todo.groupby("data")["precipitacao_mm"].apply(lambda s: bool(s.isna().all()))
    faltantes = faltantes[faltantes].index
    pd.DataFrame({"data": faltantes}).to_csv(PASTA / "dias_faltantes.csv", index=False)
    print(f"Dias sem nenhum dado no periodo: {len(faltantes)} (lista em dias_faltantes.csv)")

    for f in PASTA.glob("_parcial_v2_chuva_*.csv"):
        f.unlink()
    print("\nPronto! Arquivos gerados:")
    print(" - merge_chuva_diaria_municipios_2020_2024.csv", f"({len(todo)} linhas)")
    print(" - cdd_anual_municipios_2020_2024.csv", f"({len(cdd)} linhas)")
    print(" - centroides_municipios.csv")
    print(f"Municipios com CDD medido: {cdd['id'].nunique()} de {len(mapa)} "
          "(os demais nao tem pixel e serao imputados no passo 2).")


if __name__ == "__main__":
    main()
