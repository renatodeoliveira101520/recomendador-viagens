# Databricks notebook source
# MAGIC %md
# MAGIC # Bússola — 03 · Camada Gold (base analítica)
# MAGIC
# MAGIC Junta as **avaliações** (swipes) com o **perfil traduzido** de cada viajante:
# MAGIC **uma linha = um viajante avaliando um destino**. É a tabela usada pela EDA e pelo modelo.
# MAGIC
# MAGIC **Regras de consolidação das avaliações**
# MAGIC 1. Swipes repetidos (mesmo viajante, mesmo destino, mesma resposta) contam uma vez só.
# MAGIC 2. Se o viajante deu **respostas diferentes** para o mesmo destino (ex.: Sim e Não), o par é descartado, porque é ambíguo.
# MAGIC
# MAGIC Resultado esperado: **813.802 avaliações** · **61.130 viajantes** · **1.452 destinos**.

# COMMAND ----------

CATALOGO = "workspace"
SCHEMA   = "default"
TABELA_GOLD = "gold_stravl_v_2"      # nome usado pelos notebooks de EDA e do modelo

def tabela(nome):
    return f"{CATALOGO}.{SCHEMA}.{nome}" if CATALOGO else f"{SCHEMA}.{nome}"

# COMMAND ----------

from pyspark.sql import functions as F

viajantes = spark.table(tabela("silver_stravl_viajantes"))
destinos  = spark.table(tabela("silver_stravl_destinos"))

# 1) Uma linha por swipe
swipes = (viajantes.select("usuario_id", F.explode("swipes_sim").alias("destino_id")).withColumn("avaliacao", F.lit("Sim"))
    .unionByName(viajantes.select("usuario_id", F.explode("swipes_nao").alias("destino_id")).withColumn("avaliacao", F.lit("Não")))
    .unionByName(viajantes.select("usuario_id", F.explode("swipes_talvez").alias("destino_id")).withColumn("avaliacao", F.lit("Talvez"))))
total_swipes = swipes.count()

# 2) Consolidação: remove repetidos e descarta pares com respostas conflitantes
swipes = swipes.dropDuplicates(["usuario_id", "destino_id", "avaliacao"])
conflitos = (swipes.groupBy("usuario_id", "destino_id")
             .agg(F.countDistinct("avaliacao").alias("n"))
             .filter("n > 1").drop("n"))
avaliacoes = swipes.join(conflitos, ["usuario_id", "destino_id"], "left_anti")

# 3) Junta destino e perfil traduzido (listas gravadas como texto JSON, ex.: ["Praia","Cultura"])
PERFIL = ["form_a_pt", "form_b_pt", "form_c_pt", "form_f_pt", "form_g_pt",
          "form_h_pt", "form_i_pt", "form_j_pt", "form_r_pt", "form_rr_pt"]
perfil = viajantes.select("usuario_id", *[
    F.when(F.col(c).isNull() | (F.size(c) == 0), F.lit(None)).otherwise(F.to_json(c)).alias(c) for c in PERFIL])

gold = (avaliacoes
    .join(destinos.select("destino_id", "destino", "pais"), "destino_id")
    .join(perfil, "usuario_id")
    .withColumn("target", F.when(F.col("avaliacao") == "Sim", 1.0)
                           .when(F.col("avaliacao") == "Não", 0.0))
    .select("usuario_id", "destino", "pais", *PERFIL, "avaliacao", "target"))

gold.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(tabela(TABELA_GOLD))

# COMMAND ----------

# MAGIC %md ## Conferência

# COMMAND ----------

g = spark.table(tabela(TABELA_GOLD))
print(f"Swipes brutos:          {total_swipes:,}")
print(f"Avaliações na Gold:     {g.count():,}   (esperado 813.802)")
print(f"Viajantes:              {g.select('usuario_id').distinct().count():,}   (esperado 61.130)")
print(f"Destinos:               {g.select('destino').distinct().count():,}   (esperado 1.452)")
display(g.groupBy("avaliacao").count().orderBy(F.desc("count")))
display(g.limit(5))
