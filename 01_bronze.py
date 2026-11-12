# Databricks notebook source
# MAGIC %md
# MAGIC # Bússola — 01 · Camada Bronze (ingestão)
# MAGIC
# MAGIC Lê os **dados originais** da base pública *Stravl Travel Preference Data* e grava, **sem nenhuma transformação**,
# MAGIC duas tabelas Delta:
# MAGIC
# MAGIC | Tabela | Conteúdo |
# MAGIC |---|---|
# MAGIC | `bronze_stravl_formulario` | uma linha por viajante: respostas do formulário (códigos) e listas de swipes Sim / Não / Talvez |
# MAGIC | `bronze_stravl_destinos` | catálogo de destinos: código (posição na lista) e nome |
# MAGIC
# MAGIC Também adiciona colunas de rastreabilidade: `_fonte` e `_ingestao_ts`.
# MAGIC
# MAGIC **Fonte:** https://github.com/Stravl/stravl-data (uso acadêmico e comercial liberado).

# COMMAND ----------

# MAGIC %md ## Configuração

# COMMAND ----------

CATALOGO = "workspace"
SCHEMA   = "default"

# Fonte 1: baixar direto do GitHub (precisa de acesso à internet no Databricks)
URL_FORMULARIO = "https://raw.githubusercontent.com/Stravl/stravl-data/main/Stravl_Travel_Preference_Data.csv"
URL_DESTINOS   = "https://raw.githubusercontent.com/Stravl/stravl-data/main/destination_ids.txt"

# Fonte 2 (se a internet estiver bloqueada): suba os dois arquivos num Volume e informe a pasta aqui.
#   Catalog → workspace → default → Create → Volume (nome: stravl_raw) → Upload to this volume
PASTA_VOLUME = "/Volumes/workspace/default/stravl_raw"

def tabela(nome):
    return f"{CATALOGO}.{SCHEMA}.{nome}" if CATALOGO else f"{SCHEMA}.{nome}"

# COMMAND ----------

# MAGIC %md ## Leitura dos arquivos originais

# COMMAND ----------

import io
import urllib.request
import pandas as pd
from pyspark.sql import functions as F

def ler_texto(url, caminho_volume):
    """Tenta o GitHub; se não conseguir, lê do Volume."""
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            print("Lido do GitHub:", url)
            return r.read().decode("utf-8"), url
    except Exception as erro:
        print(f"GitHub indisponível ({erro}). Lendo do Volume: {caminho_volume}")
        with open(caminho_volume, encoding="utf-8") as f:
            return f.read(), caminho_volume

txt_form, fonte_form = ler_texto(URL_FORMULARIO, f"{PASTA_VOLUME}/Stravl_Travel_Preference_Data.csv")
txt_dest, fonte_dest = ler_texto(URL_DESTINOS, f"{PASTA_VOLUME}/destination_ids.txt")

# Formulário: tudo como texto, exatamente como veio (a Bronze não interpreta nada)
pdf_form = pd.read_csv(io.StringIO(txt_form), dtype=str, keep_default_na=False)
pdf_form = pdf_form.replace({"": None})
pdf_form.columns = [c.lower() for c in pdf_form.columns]

# Destinos: uma linha por destino; o código é a posição na lista (começando em 0)
nomes = [linha.rstrip("\r") for linha in txt_dest.split("\n") if linha.strip() != ""]
pdf_dest = pd.DataFrame({"destino_id": range(len(nomes)), "destino": nomes})

print(f"Formulário: {len(pdf_form):,} viajantes × {pdf_form.shape[1]} colunas")
print(f"Destinos no catálogo: {len(pdf_dest):,}")

# COMMAND ----------

# MAGIC %md ## Gravação das tabelas Bronze (Delta)

# COMMAND ----------

agora = F.current_timestamp()

bronze_form = (spark.createDataFrame(pdf_form)
               .withColumn("_fonte", F.lit(fonte_form))
               .withColumn("_ingestao_ts", agora))
bronze_dest = (spark.createDataFrame(pdf_dest)
               .withColumn("_fonte", F.lit(fonte_dest))
               .withColumn("_ingestao_ts", agora))

bronze_form.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(tabela("bronze_stravl_formulario"))
bronze_dest.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(tabela("bronze_stravl_destinos"))

for t in ["bronze_stravl_formulario", "bronze_stravl_destinos"]:
    print(f"{tabela(t)}: {spark.table(tabela(t)).count():,} linhas")

# COMMAND ----------

# MAGIC %md ## Conferência
# MAGIC Esperado: **80.301** viajantes e **1.486** destinos no catálogo.

# COMMAND ----------

display(spark.table(tabela("bronze_stravl_formulario")).limit(5))
