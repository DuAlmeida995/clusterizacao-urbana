# %% [markdown]
# # PASSO 2 (v2) - Tratamento dos dados, analise exploratoria e base final para clusterizacao
#
# Ordem: (1) tratamento da renda/perdas e do CDD -> (2) estatistica multianual e imputacao
# -> (3) base final com Z-score -> (4) analise exploratoria SOBRE OS DADOS JA TRATADOS
# -> (5) conferencia com K-Means.
#
# Entradas (pasta dados/):
#   cdd_anual_municipios_2020_2024.csv  (passo 1 v2: precisa da coluna dias_observados)
#   centroides_municipios.csv           (passo 1)
#   dias_consecutivos_seca.csv          (CDD 2022 da analise antiga, so para comparacao)
#   perdaDeAgua.csv, rendaPerCapitaSP.csv
# Saidas:
#   df_clusterizacao_cdd_multianual.csv <- BASE FINAL para os algoritmos
#   log_tratamento.csv                  <- cada tratamento e quantos casos afetou (para a secao de metodos)
#   figuras_cdd_multianual/*.png
#   resumo_analise_cdd_multianual.txt

# %%
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

try:
    PASTA_SCRIPT = Path(__file__).resolve().parent
except NameError:  # notebook
    PASTA_SCRIPT = Path.cwd()

ap = argparse.ArgumentParser()
ap.add_argument("--pasta", default=str(PASTA_SCRIPT))
ap.add_argument("--estatistica", default="mediana", choices=["mediana", "p90", "media", "maximo"],
                help="resumo multianual do CDD usado na clusterizacao (padrao: mediana)")
ap.add_argument("--min-frac-dias", type=float, default=0.90,
                help="um ano so entra se pelo menos essa fracao dos dias foi observada")
ap.add_argument("--min-anos", type=int, default=3,
                help="minimo de anos validos para calcular a estatistica multianual")
ap.add_argument("--fator-iqr", type=float, default=1.5)
ap.add_argument("--outliers", default="cap", choices=["cap", "remover", "nenhum"],
                help="cap: limita ao limite do IQR; remover: anula; nenhum: so registra")
args, _ = ap.parse_known_args()

PASTA = Path(args.pasta)
ANOS = [2020, 2021, 2022, 2023, 2024]
COLS = [f"cdd_{a}" for a in ANOS]
MIN_ANOS = max(2, args.min_anos)
ESCOLHIDA = {"mediana": "cdd_mediana", "p90": "cdd_p90", "media": "cdd_media", "maximo": "cdd_maximo"}[args.estatistica]
FIG = PASTA / "figuras_cdd_multianual"
FIG.mkdir(exist_ok=True)
sns.set_theme(style="whitegrid")

LOG, RESUMO = [], []


def registrar(etapa, descricao, n=None):
    LOG.append({"etapa": etapa, "descricao": descricao, "quantidade": n})
    linha = f"[{etapa}] {descricao}" + ("" if n is None else f": {n}")
    print(linha)
    RESUMO.append(linha)


def dizer(txt=""):
    print(txt)
    RESUMO.append(txt)


def limites_iqr(s, fator):
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    return q1 - fator * iqr, q3 + fator * iqr


# %% [markdown]
# ## 1. Tratamento - renda e perdas (mesmas regras do artigo, com registro de cada etapa)

# %%
def ler_renda_perdas(pasta):
    ag = pd.read_csv(pasta / "perdaDeAgua.csv", sep=";", encoding="latin1", skiprows=1, header=None,
                     names=["id_municipio", "municipio", "estado", "ano_referencia", "id_prestador",
                            "prestador", "sigla_prestador", "abrangencia", "tipo_servico",
                            "natureza_juridica", "indice_perdas_distribuicao"],
                     usecols=range(11), decimal=",")
    rd = pd.read_csv(pasta / "rendaPerCapitaSP.csv", skiprows=6, header=None,
                     names=["id_municipio", "municipio", "grupo_idade", "renda_per_capita"],
                     encoding="utf-8-sig")
    registrar("entrada", "linhas lidas - perdas (SNIS)", len(ag))
    registrar("entrada", "linhas lidas - renda (IBGE)", len(rd))

    n = len(ag)
    ag = ag.drop_duplicates()
    registrar("duplicatas", "linhas identicas removidas - perdas", n - len(ag))
    n = len(rd)
    rd = rd.drop_duplicates()
    registrar("duplicatas", "linhas identicas removidas - renda", n - len(rd))

    ag = ag.dropna(subset=["indice_perdas_distribuicao", "id_municipio"])
    rd = rd.dropna(subset=["renda_per_capita", "id_municipio"])
    for d in (ag, rd):
        d["id6"] = d["id_municipio"].astype("int64").astype(str).str[:6].astype(int)

    registrar("duplicatas", "municipios com mais de um prestador de agua (fica o primeiro)",
              int(ag["id6"].duplicated().sum()))
    ag = ag.drop_duplicates("id6")[["id6", "indice_perdas_distribuicao", "natureza_juridica"]]
    registrar("duplicatas", "municipios repetidos na renda (fica o primeiro)",
              int(rd["id6"].duplicated().sum()))
    rd = rd.drop_duplicates("id6")[["id6", "municipio", "renda_per_capita"]]
    return rd.merge(ag, on="id6", how="left")


base = ler_renda_perdas(PASTA)

n = int((base["indice_perdas_distribuicao"] == 0).sum())
base.loc[base["indice_perdas_distribuicao"] == 0, "indice_perdas_distribuicao"] = np.nan
registrar("implausiveis", "perdas exatamente 0,00% tratadas como ausentes (decisao do artigo)", n)

n = int((base["renda_per_capita"] <= 0).sum())
base.loc[base["renda_per_capita"] <= 0, "renda_per_capita"] = np.nan
registrar("implausiveis", "renda <= 0 tratada como ausente", n)

antes = len(base)
base = base.dropna(subset=["renda_per_capita", "indice_perdas_distribuicao"]).copy()
registrar("ausentes", "municipios removidos por renda ou perdas ausentes", antes - len(base))

for col in ["renda_per_capita", "indice_perdas_distribuicao"]:
    lo, hi = limites_iqr(base[col], args.fator_iqr)
    n_hi, n_lo = int((base[col] > hi).sum()), int((base[col] < lo).sum())
    base[col] = base[col].clip(lower=lo, upper=hi)
    registrar("outliers", f"{col}: acima do limite superior do IQR (limitados)", n_hi)
    registrar("outliers", f"{col}: abaixo do limite inferior do IQR (limitados)", n_lo)

# %% [markdown]
# ## 2. Tratamento - CDD multianual (MERGE/INPE)
# Ordem das regras:
# a) duplicatas (mesmo municipio e ano);
# b) anos com poucos dias observados (serie com buracos gera CDD inflado) viram ausentes;
# c) valores impossiveis (negativos ou maiores que os dias observados) viram ausentes;
# d) outliers por ano pelo criterio do IQR (cap, remover ou so registrar);
# e) a estatistica multianual exige pelo menos MIN_ANOS anos validos.

# %%
novo_path = PASTA / "cdd_anual_municipios_2020_2024.csv"
cent_path = PASTA / "centroides_municipios.csv"
antigo_path = PASTA / "dias_consecutivos_seca.csv"
for p in (novo_path, cent_path, antigo_path):
    if not p.exists():
        sys.exit(f"Arquivo ausente: {p.name}. Rode antes o passo 1 (versao 2).")

novo = pd.read_csv(novo_path)
if "dias_observados" not in novo.columns:
    sys.exit("O CSV de CDD nao tem a coluna 'dias_observados'. Rode o passo 1 na versao 2.")
novo["id6"] = novo["id"].astype(str).str[:6].astype(int)
novo["cdd"] = novo["cdd"].astype(float)
novo["cdd_bruto"] = novo["cdd"]

n = len(novo)
novo = novo.drop_duplicates()
registrar("duplicatas", "linhas identicas removidas - CDD", n - len(novo))
registrar("duplicatas", "municipio-ano repetidos com valores diferentes (fica o primeiro)",
          int(novo.duplicated(["id6", "ano"]).sum()))
novo = novo.drop_duplicates(["id6", "ano"], keep="first").copy()

dias_no_ano = np.where(novo["ano"] % 4 == 0, 366, 365)
frac = novo["dias_observados"] / dias_no_ano
mask_inc = frac < args.min_frac_dias
registrar("completude", f"municipio-ano com menos de {args.min_frac_dias:.0%} dos dias observados (anulado)",
          int(mask_inc.sum()))
novo.loc[mask_inc, "cdd"] = np.nan

mask_imp = novo["cdd"].notna() & ((novo["cdd"] < 0) | (novo["cdd"] > novo["dias_observados"]))
registrar("implausiveis", "CDD negativo ou maior que os dias observados (anulado)", int(mask_imp.sum()))
novo.loc[mask_imp, "cdd"] = np.nan

novo["flag_outlier"] = False
for ano in ANOS:
    m_ano = novo["ano"] == ano
    v = novo.loc[m_ano, "cdd"].dropna()
    if len(v) < 10:
        continue
    lo, hi = limites_iqr(v, args.fator_iqr)
    lo = max(lo, 0.0)
    alto = m_ano & (novo["cdd"] > hi)
    baixo = m_ano & (novo["cdd"] < lo)
    novo.loc[alto | baixo, "flag_outlier"] = True
    registrar("outliers", f"CDD {ano}: acima de {hi:.1f} dias (limite IQR)", int(alto.sum()))
    registrar("outliers", f"CDD {ano}: abaixo de {lo:.1f} dias (limite IQR)", int(baixo.sum()))
    if args.outliers == "cap":
        novo.loc[alto, "cdd"] = hi
        novo.loc[baixo, "cdd"] = lo
    elif args.outliers == "remover":
        novo.loc[alto | baixo, "cdd"] = np.nan

wide = novo.pivot(index="id6", columns="ano", values="cdd")
wide.columns = [f"cdd_{int(a)}" for a in wide.columns]
for c in COLS:
    if c not in wide.columns:
        wide[c] = np.nan
wide = wide[COLS]

# %% [markdown]
# ## 3. Estatistica multianual e imputacao espacial (IDW, 5 vizinhos, haversine)
# Imputa-se cada estatistica (mediana, P90, media, maximo, desvio) para os municipios
# sem dado valido, usando a media ponderada por 1/d^2 dos 5 municipios medidos mais proximos.

# %%
def haversine_km(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    a = (np.sin((lat2 - lat1) * p / 2) ** 2
         + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2)
    return 12742 * np.arcsin(np.sqrt(a))


def idw(alvo_xy, doador_xy, doador_val, k=5):
    out = []
    for lat, lon in alvo_xy:
        d = haversine_km(lat, lon, doador_xy[:, 0], doador_xy[:, 1])
        ix = np.argsort(d)[:k]
        w = 1 / np.maximum(d[ix], 1e-6) ** 2
        out.append(np.sum(doador_val[ix] * w) / w.sum())
    return np.array(out)


def calcular_estatisticas(df):
    c = df[COLS]
    out = pd.DataFrame(index=df.index)
    out["cdd_mediana"] = c.median(axis=1)
    out["cdd_p90"] = c.quantile(0.9, axis=1)
    out["cdd_media"] = c.mean(axis=1)
    out["cdd_maximo"] = c.max(axis=1)
    out["cdd_desvio"] = c.std(axis=1)
    out.loc[df["n_anos_validos"] < MIN_ANOS] = np.nan
    return out


cent = pd.read_csv(cent_path)
cent["id6"] = cent["id"].astype(str).str[:6].astype(int)
antigo = pd.read_csv(antigo_path)
antigo["id6"] = antigo["id"].astype(str).str[:6].astype(int)
antigo = antigo.drop_duplicates("id6").set_index("id6")["dias_consecutivos"].rename("cdd_2022_antigo")

tab = cent.drop_duplicates("id6").set_index("id6")[["id", "name", "lat", "lon"]].join(wide, how="left")
tab["n_anos_validos"] = tab[COLS].notna().sum(axis=1)
registrar("cobertura", f"municipios da malha com menos de {MIN_ANOS} anos validos (serao imputados)",
          int((tab["n_anos_validos"] < MIN_ANOS).sum()))
tab = tab.join(calcular_estatisticas(tab))
tab["cdd_imputado"] = tab["cdd_mediana"].isna()
tab = tab.join(antigo)

doadores = tab[tab["cdd_mediana"].notna()]
xy_doa = doadores[["lat", "lon"]].values
alvo_idx = tab.index[tab["cdd_imputado"]]

# validacao cruzada da imputacao (10 partes) no CDD escolhido
val = doadores[ESCOLHIDA].values
pred_all, true_all = [], []
for tr, te in KFold(10, shuffle=True, random_state=42).split(xy_doa):
    pred_all.append(idw(xy_doa[te], xy_doa[tr], val[tr]))
    true_all.append(val[te])
pred_all, true_all = np.concatenate(pred_all), np.concatenate(true_all)
rmse_idw = float(np.sqrt(np.mean((pred_all - true_all) ** 2)))
rmse_media = float(np.sqrt(np.mean((true_all - val.mean()) ** 2)))
corr_idw = float(np.corrcoef(pred_all, true_all)[0, 1])
dizer(f"\nImputacao validada (10 partes, {ESCOLHIDA}): RMSE IDW {rmse_idw:.1f} dias x "
      f"RMSE da media global {rmse_media:.1f} dias; correlacao {corr_idw:.2f}.")

for col in ["cdd_mediana", "cdd_p90", "cdd_media", "cdd_maximo", "cdd_desvio"]:
    if len(alvo_idx):
        tab.loc[alvo_idx, col] = idw(tab.loc[alvo_idx, ["lat", "lon"]].values, xy_doa,
                                     doadores[col].values)
registrar("imputacao", f"municipios imputados por IDW (5 vizinhos)", int(len(alvo_idx)))

# %% [markdown]
# ## 4. Base final: renda + perdas + CDD multianual, com Z-score

# %%
base = base.merge(tab.reset_index(), on="id6", how="left")
sem_malha = int(base["id"].isna().sum())
registrar("cobertura", "municipios de renda/perdas fora da malha (removidos)", sem_malha)
base = base.dropna(subset=["id"]).copy()
base["id_municipio"] = base["id"].astype("int64")
base["cdd_usado"] = base[ESCOLHIDA]

for nome, col in [("z_renda", "renda_per_capita"), ("z_perdas", "indice_perdas_distribuicao"),
                  ("z_cdd", "cdd_usado")]:
    base[nome] = StandardScaler().fit_transform(base[[col]]).ravel()
base["indice_vulnerabilidade"] = -base["z_renda"] + base["z_perdas"] + base["z_cdd"]

ordem = (["id_municipio", "municipio", "natureza_juridica", "renda_per_capita",
          "indice_perdas_distribuicao"] + COLS
         + ["n_anos_validos", "cdd_mediana", "cdd_p90", "cdd_media", "cdd_maximo", "cdd_desvio",
            "cdd_usado", "cdd_imputado", "cdd_2022_antigo", "z_renda", "z_perdas", "z_cdd",
            "indice_vulnerabilidade"])
final = base[ordem].sort_values("id_municipio").reset_index(drop=True)
assert final[["z_renda", "z_perdas", "z_cdd"]].notna().all().all(), "Ha municipios sem CDD"
final.to_csv(PASTA / "df_clusterizacao_cdd_multianual.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(LOG).to_csv(PASTA / "log_tratamento.csv", index=False, encoding="utf-8-sig")
dizer(f"\nBase final: {len(final)} municipios; CDD usado = {ESCOLHIDA}; "
      f"imputados: {int(final['cdd_imputado'].sum())}.")

# %% [markdown]
# ## 5. Analise exploratoria (sobre os dados JA tratados)

# %%
# 5.1 efeito do tratamento no CDD
bruto = novo["cdd_bruto"].dropna()
trat = novo["cdd"].dropna()
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(bruto, bins=40, ax=ax[0], color="#4c78a8")
ax[0].set_title(f"CDD bruto (n = {len(bruto)})")
sns.histplot(trat, bins=40, ax=ax[1], color="#f58518")
ax[1].set_title(f"CDD apos tratamento (n = {len(trat)})")
plt.tight_layout()
plt.savefig(FIG / "cdd_antes_depois_tratamento.png", dpi=150)
plt.close()
dizer(f"\nCDD bruto: max {bruto.max():.0f} dias; apos tratamento: max {trat.max():.0f} dias.")

# 5.2 descritivas por ano (tratado)
medidos = tab[tab["n_anos_validos"] >= MIN_ANOS][COLS]
dizer("\n--- CDD por ano, tratado (municipios com dado valido) ---")
dizer(medidos.describe().T[["count", "mean", "50%", "std", "min", "max"]].round(1).to_string())

# 5.3 consistencia entre anos
corr = medidos.corr(method="spearman", min_periods=50)
dizer("\n--- Spearman entre os anos (tratado) ---")
dizer(corr.round(2).to_string())

completos = medidos.dropna()
dp = completos.std(axis=1)
cv = dp / completos.mean(axis=1)
dizer(f"\nMunicipios com os 5 anos validos: {len(completos)}. Desvio medio dentro do municipio: "
      f"{dp.mean():.1f} dias; CV medio {cv.mean():.1%}.")
dizer("Ano com a maior seca de cada municipio: "
      + ", ".join(f"{k[-4:]}={v}" for k, v in completos.idxmax(axis=1).value_counts().sort_index().items()))

fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
sns.boxplot(data=medidos, ax=ax[0], color="#4c78a8")
ax[0].set_xticks(range(len(ANOS)), ANOS)
ax[0].set_title("CDD anual por ano (tratado)")
ax[0].set_ylabel("Dias")
sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1, ax=ax[1])
ax[1].set_title("Correlacao de Spearman entre anos")
plt.tight_layout()
plt.savefig(FIG / "cdd_por_ano_tratado.png", dpi=150)
plt.close()

# 5.4 consistencia com a analise antiga (2022)
comp = tab[["cdd_2022", "cdd_2022_antigo"]].dropna()
dif = comp["cdd_2022"] - comp["cdd_2022_antigo"]
dizer(f"\nCDD 2022 tratado x antigo: n = {len(comp)}; iguais {(dif == 0).mean():.1%}; "
      f"correlacao {comp.corr().iloc[0, 1]:.3f}; diferenca media {dif.mean():+.2f} dias.")
fig, ax = plt.subplots(figsize=(5.5, 5.5))
ax.scatter(comp["cdd_2022_antigo"], comp["cdd_2022"], s=12, alpha=0.5)
lim = [0, max(comp.max()) + 3]
ax.plot(lim, lim, "r--", lw=1)
ax.set(xlabel="CDD 2022 - analise antiga", ylabel="CDD 2022 - tratado", title="Conferencia de 2022")
plt.tight_layout()
plt.savefig(FIG / "cdd_2022_tratado_vs_antigo.png", dpi=150)
plt.close()

# 5.5 variavel usada na clusterizacao
dizer(f"\n--- {ESCOLHIDA} (base final) ---")
dizer(final["cdd_usado"].describe().round(2).to_string())
rho_r = spearmanr(final["renda_per_capita"], final["cdd_usado"])[0]
rho_p = spearmanr(final["indice_perdas_distribuicao"], final["cdd_usado"])[0]
dizer(f"Spearman do CDD usado com renda: {rho_r:+.2f}; com perdas: {rho_p:+.2f} "
      "(dimensoes pouco correlacionadas favorecem a abordagem multivariada)")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
sns.histplot(final["cdd_usado"], bins=25, kde=True, ax=ax[0], color="#4c78a8")
ax[0].set_title(f"Distribuicao de {ESCOLHIDA}")
sns.histplot(final["cdd_desvio"].dropna(), bins=25, ax=ax[1], color="#f58518")
ax[1].set_title("Desvio padrao entre anos (dias)")
plt.tight_layout()
plt.savefig(FIG / "cdd_base_final.png", dpi=150)
plt.close()

# %% [markdown]
# ## 6. Conferencia com K-Means (k = 2..10)
# Compara o agrupamento com o CDD de 2022 (analise antiga) e com o CDD multianual tratado.
# Serve para checar a base; a analise completa de perfis continua nos notebooks.

# %%
def varredura_kmeans(X, ks=range(2, 11)):
    res = {}
    for k in ks:
        km = KMeans(k, init="k-means++", n_init=50, random_state=42).fit(X)
        res[k] = (silhouette_score(X, km.labels_), km.labels_)
    return res


X_multi = final[["z_renda", "z_perdas", "z_cdd"]].values
cdd22 = final["cdd_2022"].fillna(final["cdd_usado"])
z22 = StandardScaler().fit_transform(cdd22.to_frame()).ravel()
X_2022 = np.column_stack([final["z_renda"], final["z_perdas"], z22])

r_multi = varredura_kmeans(X_multi)
r_2022 = varredura_kmeans(X_2022)
tab_sil = pd.DataFrame({"k": list(r_multi),
                        "silhouette_cdd_2022": [round(r_2022[k][0], 3) for k in r_multi],
                        "silhouette_cdd_multianual": [round(r_multi[k][0], 3) for k in r_multi]})
dizer("\n--- Silhouette por k ---")
dizer(tab_sil.to_string(index=False))
ari = adjusted_rand_score(r_2022[4][1], r_multi[4][1])
dizer(f"\nk=4: ARI entre CDD 2022 e CDD multianual = {ari:.2f}")
tab_sil.to_csv(PASTA / "comparacao_silhouette_cdd.csv", index=False)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(tab_sil["k"], tab_sil["silhouette_cdd_2022"], "o-", label="CDD 2022")
ax.plot(tab_sil["k"], tab_sil["silhouette_cdd_multianual"], "s-", label=f"CDD multianual ({args.estatistica})")
ax.set(xlabel="k", ylabel="Silhouette medio", title="Silhouette por numero de clusters")
ax.legend()
plt.tight_layout()
plt.savefig(FIG / "silhouette_comparacao.png", dpi=150)
plt.close()

(PASTA / "resumo_analise_cdd_multianual.txt").write_text("\n".join(RESUMO), encoding="utf-8")
print("\nConcluido. Veja df_clusterizacao_cdd_multianual.csv, log_tratamento.csv e figuras_cdd_multianual/")
