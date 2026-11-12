# Databricks notebook source
# MAGIC %md
# MAGIC # Bússola — 04 · Análise Exploratória (EDA) da camada Gold
# MAGIC Responde às 5 perguntas da EDA do projeto:
# MAGIC 1. Perfil e preferências dos viajantes · 2. Distribuição Sim / Não / Talvez · 3. Destinos com mais avaliações e suas taxas de aceitação
# MAGIC 4. Relação entre orçamento, estação, experiências e interesse · 5. Combinações de preferências (personas)
# MAGIC
# MAGIC ➡️ Ajuste só o nome da tabela na célula de configuração e rode tudo (Run all).

# COMMAND ----------

# MAGIC %pip install pycountry-convert

# COMMAND ----------

TABELA_GOLD = "workspace.default.gold_stravl_v_2"

# COMMAND ----------

import json
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import pycountry_convert as pc
from sklearn.cluster import KMeans

def mostrar(tabela, titulo):
    """Mostra a tabela no notebook (no Databricks, use o botão + da tabela para virar gráfico)."""
    print("\n" + "=" * 8, titulo)
    display(tabela.round(4) if hasattr(tabela, "round") else tabela)
    return tabela

df = spark.table(TABELA_GOLD).toPandas()
print(df.shape)


# COMMAND ----------

FORMS = {"form_a_pt": "Faixa etária", "form_b_pt": "Orçamento", "form_c_pt": "Estação",
         "form_f_pt": "Experiência", "form_g_pt": "Paisagem"}
ORDEM = {"form_a_pt": ["0-19 anos", "20-39 anos", "40-59 anos", "60+ anos"],
         "form_b_pt": ["0-49", "50-99", "100-249", "300+"],
         "form_c_pt": ["Verão", "Primavera", "Outono", "Inverno"]}
lista = lambda s: json.loads(s) if isinstance(s, str) else []
for c in FORMS:
    df[c + "_l"] = df[c].map(lista)

ALL_B = '["0-49","50-99","100-249","300+"]'
df["resposta_padrao"] = df["form_b_pt"] == ALL_B   # marcou todas as faixas de orçamento (e de idade)

# COMMAND ----------

# MAGIC %md ## continente / hemisfério do destino

# COMMAND ----------

CARIBE = {"Bahamas","Barbados","Jamaica","Dominican Republic","Cuba","Haiti","Trinidad and Tobago","Aruba",
          "Curacao","Curaçao","Saint Lucia","St. Lucia","Grenada","Antigua and Barbuda","Dominica","Cayman Islands",
          "Turks and Caicos Islands","St. Vincent and the Grenadines","Saint Kitts and Nevis","Bermuda",
          "Puerto Rico","US Virgin Islands","British Virgin Islands","Martinique","Guadeloupe","Sint Maarten",
          "Bonaire","Anguilla","Montserrat","Saint Barthélemy","Saint Martin"}
NOME_CONT = {"EU": "Europa", "NA": "América do Norte", "SA": "América do Sul", "AS": "Ásia",
             "AF": "África", "OC": "Oceania"}
ORIENTE_MEDIO = {"Turkey","Israel","Jordan","United Arab Emirates","Qatar","Oman","Saudi Arabia","Lebanon",
                 "Bahrain","Kuwait","Iran","Iraq","Egypt","Cyprus"}
def continente(p, destino):
    if p in CARIBE or "Puerto Rico" in destino or "Virgin Islands" in destino: return "Caribe"
    if p in ORIENTE_MEDIO and p not in ("Cyprus","Turkey"): return "Oriente Médio"
    if p == "Greenland (Denmark)": return "América do Norte"
    try: return NOME_CONT[pc.country_alpha2_to_continent_code(pc.country_name_to_country_alpha2(p))]
    except Exception: return "Outro"
SUL = {"Brazil","Argentina","Chile","Uruguay","Paraguay","Bolivia","Peru","Australia","New Zealand","South Africa",
       "Namibia","Botswana","Zimbabwe","Mozambique","Madagascar","Mauritius","Fiji","French Polynesia","Indonesia",
       "Tanzania","Zambia","Malawi","Seychelles","Reunion","Samoa","Tonga","Cook Islands","Vanuatu","New Caledonia"}
dest_pais = df.drop_duplicates("destino")[["destino", "pais"]]
dest_pais["continente"] = [continente(p, d) for p, d in zip(dest_pais.pais, dest_pais.destino)]
dest_pais["hemisferio"] = np.where(dest_pais.pais.isin(SUL), "Sul", "Norte")
df = df.merge(dest_pais[["destino", "continente", "hemisferio"]], on="destino")

r = df[df.target.notna()].copy()           # Sim/Não
u = df.drop_duplicates("usuario_id")
res = {}

# COMMAND ----------

# MAGIC %md ## 1. visão geral

# COMMAND ----------

n_user = df.groupby("usuario_id").size()
res["geral"] = {
    "linhas": int(len(df)), "viajantes": int(df.usuario_id.nunique()), "destinos": int(df.destino.nunique()),
    "paises": int(df.pais.nunique()), "colunas": int(len([c for c in df.columns if not c.endswith('_l')
                                                          and c not in ('resposta_padrao','continente','hemisferio')])),
    "duplicadas_usuario_destino": int(df.duplicated(["usuario_id", "destino"]).sum()),
    "aval_por_viajante_mediana": float(n_user.median()),
    "viajantes_exatamente_10": int((n_user == 10).sum()),
    "viajantes_mais_50": int((n_user > 50).sum()),
    "linhas_viajantes_mais_50": int(n_user[n_user > 50].sum()),
    "max_aval_viajante": int(n_user.max()),
    "viajantes_resposta_padrao": int(u.resposta_padrao.sum()),
}
nul = {FORMS[c]: int(u[c].isna().sum()) for c in FORMS}
res["nulos_por_viajante"] = nul

aval = df.avaliacao.value_counts().rename_axis("avaliacao").reset_index(name="qtd")
aval["pct"] = aval.qtd / aval.qtd.sum()
aval.pipe(mostrar, "01 avaliacoes")

faixas = pd.cut(n_user, [0, 9, 10, 20, 50, 10_000], labels=["1-9", "10", "11-20", "21-50", "51+"])
faixas.value_counts().reindex(["1-9", "10", "11-20", "21-50", "51+"]).rename_axis("avaliacoes_por_viajante") \
    .reset_index(name="viajantes").pipe(mostrar, "02 aval_por_viajante")

# COMMAND ----------

# MAGIC %md ## 2. perfil dos viajantes

# COMMAND ----------

linhas = []
for c, nome in FORMS.items():
    s = u[c + "_l"].explode().dropna().value_counts()
    for v, q in s.items():
        linhas.append({"pergunta": nome, "resposta": v, "viajantes": int(q), "pct_viajantes": q / len(u)})
perfil = pd.DataFrame(linhas)
perfil.pipe(mostrar, "03 perfil_respostas")

qtd_sel = pd.DataFrame({
    "qtd_opcoes": range(0, 9),
    "Experiência": [int((u.form_f_pt_l.map(len) == k).sum()) for k in range(9)],
    "Paisagem": [int((u.form_g_pt_l.map(len) == k).sum()) for k in range(9)]})
qtd_sel.pipe(mostrar, "04 qtd_opcoes_marcadas")

# coocorrência de experiências (% dos que marcaram a linha que também marcaram a coluna)
exps = ["Praia","Aventura","Natureza","Cultura","Vida noturna","História","Compras","Gastronomia"]
M = pd.DataFrame({e: u.form_f_pt_l.map(lambda l, e=e: e in l) for e in exps})
co = pd.DataFrame({b: [M.loc[M[a], b].mean() for a in exps] for b in exps}, index=exps)
co.rename_axis("experiencia").reset_index().pipe(mostrar, "05 coocorrencia_experiencias")

# COMMAND ----------

# MAGIC %md ## 3. comportamento de avaliação

# COMMAND ----------

ur = r.groupby("usuario_id").target.mean()
dist = pd.cut(ur, [-0.01, 0, 0.2, 0.4, 0.6, 0.8, 0.999, 1],
              labels=["0% (só Não)", "1-20%", "21-40%", "41-60%", "61-80%", "81-99%", "100% (só Sim)"])
dist.value_counts().reindex(dist.cat.categories).rename_axis("taxa_sim_do_viajante") \
    .reset_index(name="viajantes").pipe(mostrar, "06 taxa_sim_por_viajante")
res["viajantes_so_nao"] = int((ur == 0).sum()); res["viajantes_so_sim"] = int((ur == 1).sum())
res["viajantes_com_sim_nao"] = int(len(ur))
res["taxa_sim_global"] = float(r.target.mean())

pad = r.groupby("resposta_padrao").target.agg(["mean", "size"])
res["taxa_sim_resposta_padrao"] = float(pad.loc[True, "mean"]); res["taxa_sim_demais"] = float(pad.loc[False, "mean"])

# COMMAND ----------

# MAGIC %md ## 4. destinos

# COMMAND ----------

d = r.groupby("destino").target.agg(taxa_sim="mean", avaliacoes="size").reset_index() \
    .merge(dest_pais, on="destino")
d.pipe(mostrar, "07 destinos")
res["destinos_menos_30"] = int((d.avaliacoes < 30).sum())
res["aval_destino_mediana"] = float(d.avaliacoes.median())
d.sort_values("avaliacoes", ascending=False).head(15).pipe(mostrar, "08 top_volume")
big = d[d.avaliacoes >= 300]
big.sort_values("taxa_sim", ascending=False).head(15).pipe(mostrar, "09 maior_aceitacao")
big.sort_values("taxa_sim").head(15).pipe(mostrar, "10 menor_aceitacao")

cont = r.groupby("continente").agg(avaliacoes=("target", "size"), taxa_sim=("target", "mean"),
                                   destinos=("destino", "nunique")).reset_index()
cont["pct_avaliacoes"] = cont.avaliacoes / cont.avaliacoes.sum()
cont.sort_values("avaliacoes", ascending=False).pipe(mostrar, "11 continentes")
paises = r.groupby("pais").agg(avaliacoes=("target", "size"), taxa_sim=("target", "mean")).reset_index()
paises.sort_values("avaliacoes", ascending=False).head(15).pipe(mostrar, "12 paises")
us = r.pais.eq("United States")
res["pct_aval_eua"] = float(us.mean()); res["taxa_sim_eua"] = float(r[us].target.mean())
res["taxa_sim_fora_eua"] = float(r[~us].target.mean())

# COMMAND ----------

# MAGIC %md ## 5. preferências × interesse

# COMMAND ----------

def taxa_por(c):
    e = r[["target", c + "_l"]].explode(c + "_l").dropna()
    t = e.groupby(c + "_l").target.agg(taxa_sim="mean", avaliacoes="size").reset_index() \
        .rename(columns={c + "_l": "resposta"})
    t.insert(0, "pergunta", FORMS[c]); return t
pd.concat([taxa_por(c) for c in FORMS]).pipe(mostrar, "13 taxa_sim_por_resposta")

# estação × hemisfério do destino (só quem marcou uma estação)
est = r[r.form_c_pt_l.map(len) == 1].assign(estacao=lambda x: x.form_c_pt_l.str[0])
eh = est.groupby(["estacao", "hemisferio"]).target.mean().unstack().reindex(ORDEM["form_c_pt"])
eh.reset_index().pipe(mostrar, "14 estacao_x_hemisferio")

# orçamento × continente
orc = r[r.form_b_pt_l.map(len) == 1].assign(orcamento=lambda x: x.form_b_pt_l.str[0])
oc = orc.groupby(["orcamento", "continente"]).target.mean().unstack().reindex(ORDEM["form_b_pt"])
oc.reset_index().pipe(mostrar, "15 orcamento_x_continente")

# experiência × continente (taxa de Sim)
ex = r[["target", "continente", "form_f_pt_l"]].explode("form_f_pt_l").dropna()
ec = ex.groupby(["form_f_pt_l", "continente"]).target.mean().unstack().reindex(exps)
ec.rename_axis("experiencia").reset_index().pipe(mostrar, "16 experiencia_x_continente")

# lift: destinos que um grupo aprova muito mais que o resto
base_d = r.groupby("destino").target.agg(["sum", "size"])
def lift_destinos(mask, rotulo, minimo=100, top=8):
    g = r[mask].groupby("destino").target.agg(["sum", "size"])
    resto = base_d.sub(g, fill_value=0)
    t = pd.DataFrame({"taxa_grupo": g["sum"] / g["size"], "n_grupo": g["size"],
                      "taxa_resto": resto["sum"] / resto["size"]}).dropna()
    t = t[(t.n_grupo >= minimo) & (resto.loc[t.index, "size"] >= minimo)]
    t["diferenca_pp"] = (t.taxa_grupo - t.taxa_resto) * 100
    t = t.sort_values("diferenca_pp", ascending=False).head(top).reset_index()
    t.insert(0, "grupo", rotulo); return t
lifts = [lift_destinos(r.form_f_pt_l.map(lambda l, e=e: e in l), f"Experiência: {e}") for e in exps]
lifts += [lift_destinos(r.form_g_pt_l.map(lambda l, e=e: e in l), f"Paisagem: {e}")
          for e in ["Mar", "Montanha", "Urbana", "Deserto", "Selva"]]
lifts += [lift_destinos(r.form_c_pt_l.map(lambda l, e=e: l == [e]), f"Estação: {e}") for e in ["Verão", "Inverno"]]
lifts += [lift_destinos(r.form_b_pt_l.map(lambda l, e=e: l == [e]), f"Orçamento: {e}") for e in ["0-49", "300+"]]
pd.concat(lifts).pipe(mostrar, "17 destinos_por_preferencia")

# o ranking de destinos muda conforme a preferência? (correlação de Spearman entre taxas)
sp = []
for e in exps:
    m = r.form_f_pt_l.map(lambda l, e=e: e in l)
    a = r[m].groupby("destino").target.agg(["mean", "size"]); b = r[~m].groupby("destino").target.agg(["mean", "size"])
    j = a.join(b, lsuffix="_g", rsuffix="_r"); j = j[(j.size_g >= 50) & (j.size_r >= 50)]
    sp.append({"experiencia": e, "spearman_ranking": j.mean_g.corr(j.mean_r, method="spearman"), "destinos": len(j)})
pd.DataFrame(sp).pipe(mostrar, "18 spearman_ranking")

# COMMAND ----------

# MAGIC %md ## 6. personas (clusters de perfis)

# COMMAND ----------

uu = u[~u.resposta_padrao].copy()
feats = {}
for c in FORMS:
    for v in sorted(set(uu[c + "_l"].explode().dropna())):
        feats[f"{FORMS[c]}: {v}"] = uu[c + "_l"].map(lambda l, v=v: v in l).astype(float)
X = pd.DataFrame(feats)
km = KMeans(n_clusters=5, n_init=10, random_state=42).fit(X.values)
uu["persona"] = km.labels_
X["persona"] = km.labels_
perf = X.groupby("persona").mean()
desc = []
r2 = r.merge(uu[["usuario_id", "persona"]], on="usuario_id")
for k in range(5):
    p = perf.loc[k]
    destaque = (p - X.drop(columns="persona").mean()).sort_values(ascending=False).head(5)
    top_d = lift_destinos(r.usuario_id.isin(uu.usuario_id[uu.persona == k]), f"Persona {k+1}", minimo=60, top=5)
    def moda(pref):
        cols = [c for c in p.index if c.startswith(pref)]
        return p[cols].idxmax().split(": ")[1]
    desc.append({"persona": k + 1, "viajantes": int((uu.persona == k).sum()),
                 "pct": float((uu.persona == k).mean()),
                 "orcamento_mais_comum": moda("Orçamento"), "estacao_mais_comum": moda("Estação"),
                 "idade_mais_comum": moda("Faixa etária"),
                 "traços_que_mais_destacam": "; ".join(f"{i} ({p[i]:.0%})" for i in destaque.index),
                 "taxa_sim": float(r2[r2.persona == k].target.mean()),
                 "destinos_que_mais_aprova_vs_resto": "; ".join(top_d.destino.head(5))})
pd.DataFrame(desc).pipe(mostrar, "19 personas")

print(json.dumps(res, ensure_ascii=False, indent=1))

# COMMAND ----------

# MAGIC %md ## Gráficos principais

# COMMAND ----------

fig, ax = plt.subplots(1, 3, figsize=(18, 4.5))
aval.set_index("avaliacao")["qtd"].plot.bar(ax=ax[0], color=["#c0392b", "#27ae60", "#f39c12"]); ax[0].set_title("Avaliações")
dist.value_counts().reindex(dist.cat.categories).plot.bar(ax=ax[1], color="#2c7fb8"); ax[1].set_title("Viajantes por taxa de Sim")
cont.set_index("continente").sort_values("taxa_sim")["taxa_sim"].plot.barh(ax=ax[2], color="#2c7fb8"); ax[2].set_title("Taxa de Sim por continente do destino")
plt.tight_layout(); plt.show()

fig, ax = plt.subplots(1, 2, figsize=(16, 4.5))
p = perfil[perfil.pergunta == "Experiência"].sort_values("pct_viajantes")
ax[0].barh(p.resposta, p.pct_viajantes, color="#2c7fb8"); ax[0].set_title("Experiências desejadas (% dos viajantes)")
p = perfil[perfil.pergunta == "Paisagem"].sort_values("pct_viajantes")
ax[1].barh(p.resposta, p.pct_viajantes, color="#41ab5d"); ax[1].set_title("Paisagens desejadas (% dos viajantes)")
plt.tight_layout(); plt.show()

# COMMAND ----------

print(json.dumps(res, ensure_ascii=False, indent=1))
