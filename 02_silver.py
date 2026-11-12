# Databricks notebook source
# MAGIC %md
# MAGIC # Bússola — 02 · Camada Silver (tratamento e qualidade)
# MAGIC
# MAGIC A partir da Bronze:
# MAGIC 1. **Converte** as respostas (texto `"['0', '1']"`) em listas.
# MAGIC 2. **Traduz** os códigos para português, conforme o dicionário de dados.
# MAGIC 3. **Cria flags de qualidade** — os registros são preservados e sinalizados, nunca apagados.
# MAGIC 4. **Enriquece o catálogo de destinos** com país e região.
# MAGIC
# MAGIC | Tabela | Conteúdo |
# MAGIC |---|---|
# MAGIC | `silver_stravl_viajantes` | uma linha por viajante: respostas traduzidas, listas de swipes e flags |
# MAGIC | `silver_stravl_destinos` | código, nome, país e região de cada destino |

# COMMAND ----------

CATALOGO = "workspace"
SCHEMA   = "default"

def tabela(nome):
    return f"{CATALOGO}.{SCHEMA}.{nome}" if CATALOGO else f"{SCHEMA}.{nome}"

# COMMAND ----------

# MAGIC %md ## Dicionário de dados (códigos → português)

# COMMAND ----------

DICIONARIO = {
    "form_a":  {"0": "0-19 anos", "1": "20-39 anos", "2": "40-59 anos", "3": "60+ anos"},           # faixas etárias do grupo
    "form_b":  {"0": "0-49", "1": "50-99", "2": "100-249", "3": "300+"},                          # orçamento US$/pessoa/noite
    "form_c":  {"0": "Inverno", "1": "Primavera", "2": "Verão", "3": "Outono"},                     # estação da viagem
    "form_f":  {"0": "Praia", "1": "Aventura", "2": "Natureza", "3": "Cultura", "4": "Vida noturna",
                "5": "História", "6": "Compras", "7": "Gastronomia"},                               # experiências
    "form_g":  {"0": "Urbana", "1": "Rural", "2": "Mar", "3": "Montanha", "4": "Lago", "5": "Deserto",
                "6": "Planícies", "7": "Selva"},                                                    # paisagens
    "form_h":  {"0": "Relaxado e tranquilo", "1": "Equilibrado", "2": "Ativo"},                     # nível de atividade
    "form_i":  {"0": "Muito preocupado com segurança", "1": "Equilibrado", "2": "Preparado para tudo"},  # segurança
    "form_j":  {"0": "Fora dos caminhos comuns", "1": "Local clássico", "2": "Popular e moderno"},  # popularidade
    "form_r":  {"0": "Em qualquer lugar", "1": "Regiões específicas"},                              # para onde quer ir
    "form_rr": {"e": "Europa", "n": "América do Norte", "c": "Caribe", "a": "Ásia", "s": "América do Sul",
                "m": "Oriente Médio", "f": "África", "o": "Oceania"},                               # regiões
}
# form_k não consta no dicionário oficial do Stravl, por isso fica de fora.

# COMMAND ----------

# MAGIC %md ## Viajantes: listas, tradução e flags

# COMMAND ----------

from itertools import chain
from pyspark.sql import functions as F

bronze = spark.table(tabela("bronze_stravl_formulario"))

def lista_texto(col):
    # "['0', '1']"  ->  ["0", "1"]
    return F.from_json(F.regexp_replace(F.col(col), "'", '"'), "array<string>")

def lista_inteiros(col):
    # "[837, 779]"  ->  [837, 779]
    return F.from_json(F.col(col), "array<int>")

def traduzir(col, mapa):
    m = F.create_map(*[F.lit(x) for x in chain(*mapa.items())])
    return F.transform(lista_texto(col), lambda x: F.element_at(m, x))

silver = bronze.select(
    F.col("id").alias("usuario_id"),
    *[traduzir(c, mapa).alias(f"{c}_pt") for c, mapa in DICIONARIO.items()],
    F.coalesce(lista_inteiros("yes_swipes"), F.array().cast("array<int>")).alias("swipes_sim"),
    F.coalesce(lista_inteiros("no_swipes"), F.array().cast("array<int>")).alias("swipes_nao"),
    F.coalesce(lista_inteiros("maybe_swipes"), F.array().cast("array<int>")).alias("swipes_talvez"),
    F.col("_ingestao_ts"),
)

# Flags de qualidade (sinalizam, não removem)
silver = (silver
    .withColumn("flag_orcamento_todos",    F.coalesce(F.size("form_b_pt") == 4, F.lit(False)))
    .withColumn("flag_idade_todas",        F.coalesce(F.size("form_a_pt") == 4, F.lit(False)))
    .withColumn("flag_popularidade_todas", F.coalesce(F.size("form_j_pt") == 3, F.lit(False)))
    .withColumn("flag_geo_inconsistente",
                ((F.array_contains("form_r_pt", "Em qualquer lugar") & F.col("form_rr_pt").isNotNull()) |
                 (F.array_contains("form_r_pt", "Regiões específicas") & F.col("form_rr_pt").isNull()))
                .cast("boolean"))
    .withColumn("flag_geo_inconsistente", F.coalesce("flag_geo_inconsistente", F.lit(False)))
    .withColumn("flag_formulario_incompleto",
                F.col("form_a_pt").isNull() | F.col("form_b_pt").isNull() | F.col("form_c_pt").isNull() |
                F.col("form_f_pt").isNull() | F.col("form_g_pt").isNull())
    .withColumn("flag_sem_avaliacoes",
                (F.size("swipes_sim") + F.size("swipes_nao") + F.size("swipes_talvez")) == 0)
)

silver.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(tabela("silver_stravl_viajantes"))

# COMMAND ----------

# MAGIC %md ## Destinos: país e região
# MAGIC O país é o último trecho do nome ("Rio de Janeiro, Brazil" → "Brazil"). A região usa as mesmas categorias da pergunta RR.

# COMMAND ----------

REGIAO_POR_PAIS = {
    "Albania": "Europa", "Algeria": "África", "Andorra": "Europa", "Antigua and Barbuda": "Caribe",
    "Argentina": "América do Sul", "Armenia": "Europa", "Australia": "Oceania", "Austria": "Europa",
    "Azerbaijan": "Europa", "Bahamas": "Caribe", "Bahrain": "Oriente Médio", "Barbados": "Caribe",
    "Belarus": "Europa", "Belgium": "Europa", "Belize": "América do Norte", "Bolivia": "América do Sul",
    "Bosnia and Herzegovina": "Europa", "Botswana": "África", "Brazil": "América do Sul", "Brunei": "Ásia",
    "Bulgaria": "Europa", "Canada": "América do Norte", "Cape Verde": "África", "Chile": "América do Sul",
    "China": "Ásia", "Colombia": "América do Sul", "Costa Rica": "América do Norte", "Croatia": "Europa",
    "Cuba": "Caribe", "Cyprus": "Europa", "Czech Republic": "Europa", "Denmark": "Europa", "Dominica": "Caribe",
    "Dominican Republic": "Caribe", "Ecuador": "América do Sul", "Egypt": "Oriente Médio",
    "El Salvador": "América do Norte", "Estonia": "Europa", "Ethiopia": "África", "Fiji": "Oceania",
    "Finland": "Europa", "France": "Europa", "Georgia": "Europa", "Germany": "Europa", "Greece": "Europa",
    "Greenland (Denmark)": "América do Norte", "Grenada": "Caribe", "Guatemala": "América do Norte",
    "Guyana": "América do Sul", "Honduras": "América do Norte", "Hungary": "Europa", "Iceland": "Europa",
    "India": "Ásia", "Indonesia": "Ásia", "Ireland": "Europa", "Israel": "Oriente Médio", "Italy": "Europa",
    "Jamaica": "Caribe", "Japan": "Ásia", "Jordan": "Oriente Médio", "Kazakhstan": "Ásia", "Kenya": "África",
    "Kuwait": "Oriente Médio", "Kyrgyzstan": "Ásia", "Laos": "Ásia", "Latvia": "Europa", "Lebanon": "Oriente Médio",
    "Liechtenstein": "Europa", "Lithuania": "Europa", "Luxembourg": "Europa", "Madagascar": "África",
    "Malaysia": "Ásia", "Maldives": "Ásia", "Malta": "Europa", "Marshall Islands": "Oceania", "Mauritius": "África",
    "Mexico": "América do Norte", "Micronesia": "Oceania", "Moldova": "Europa", "Monaco": "Europa",
    "Montenegro": "Europa", "Morocco": "África", "Mozambique": "África", "Myanmar": "Ásia", "Namibia": "África",
    "Nauru": "Oceania", "Nepal": "Ásia", "Netherlands": "Europa", "New Zealand": "Oceania",
    "Nicaragua": "América do Norte", "Norway": "Europa", "Oman": "Oriente Médio", "Palau": "Oceania",
    "Palestine": "Oriente Médio", "Panama": "América do Norte", "Papua New Guinea": "Oceania",
    "Paraguay": "América do Sul", "Peru": "América do Sul", "Philippines": "Ásia", "Poland": "Europa",
    "Portugal": "Europa", "Qatar": "Oriente Médio", "Romania": "Europa", "Russia": "Europa", "Samoa": "Oceania",
    "San Marino": "Europa", "Sao Tome and Principe": "África", "Saudi Arabia": "Oriente Médio", "Senegal": "África",
    "Serbia": "Europa", "Seychelles": "África", "Singapore": "Ásia", "Slovakia": "Europa", "Slovenia": "Europa",
    "Solomon Islands": "Oceania", "South Africa": "África", "South Korea": "Ásia", "Spain": "Europa",
    "Sri Lanka": "Ásia", "St. Kitts and Nevis": "Caribe", "St. Lucia": "Caribe",
    "St. Vincent and the Grenadines": "Caribe", "Suriname": "América do Sul", "Sweden": "Europa",
    "Switzerland": "Europa", "Taiwan": "Ásia", "Tajikistan": "Ásia", "Tanzania": "África", "Thailand": "Ásia",
    "Trinidad and Tobago": "Caribe", "Tunisia": "África", "Turkey": "Europa", "Uganda": "África",
    "Ukraine": "Europa", "United Arab Emirates": "Oriente Médio", "United Kingdom": "Europa",
    "United States": "América do Norte", "Uruguay": "América do Sul", "Uzbekistan": "Ásia", "Vanuatu": "Oceania",
    "Vietnam": "Ásia", "Zambia": "África", "Zimbabwe": "África",
}
mapa_regiao = F.create_map(*[F.lit(x) for x in chain(*REGIAO_POR_PAIS.items())])

destinos = (spark.table(tabela("bronze_stravl_destinos"))
    .withColumn("pais", F.trim(F.element_at(F.split("destino", ","), -1)))
    .withColumn("regiao",
        F.when(F.col("destino").contains("Puerto Rico") | F.col("destino").contains("Virgin Islands"), "Caribe")
         .when(F.col("destino").contains("Northern Mariana") | F.col("destino").contains("Guam"), "Oceania")
         .otherwise(F.coalesce(F.element_at(mapa_regiao, F.col("pais")), F.lit("Outra"))))
    .select("destino_id", "destino", "pais", "regiao", "_ingestao_ts"))

destinos.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(tabela("silver_stravl_destinos"))

# COMMAND ----------

# MAGIC %md ## Relatório de qualidade
# MAGIC Esperado: 80.301 viajantes · 17.625 com todas as faixas de orçamento · 17.623 com todas as opções de popularidade · 413 com inconsistência geográfica.

# COMMAND ----------

sv = spark.table(tabela("silver_stravl_viajantes"))
flags = [c for c in sv.columns if c.startswith("flag_")]
display(sv.select(F.count("*").alias("viajantes"), *[F.sum(F.col(c).cast("int")).alias(c) for c in flags]))
display(spark.table(tabela("silver_stravl_destinos")).groupBy("regiao").count().orderBy(F.desc("count")))
