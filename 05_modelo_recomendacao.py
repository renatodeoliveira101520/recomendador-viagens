# Databricks notebook source
# MAGIC %md
# MAGIC # Bússola — 05 · Modelo de Recomendação de Destinos
# MAGIC
# MAGIC **O que este notebook faz:**
# MAGIC 1. Lê a tabela **Gold** (uma linha = um viajante avaliando um destino).
# MAGIC 2. Treina uma **Regressão Logística** que estima a probabilidade de o viajante dizer **"Sim"** para cada destino.
# MAGIC 3. Avalia o modelo em viajantes que ele **nunca viu** (separação treino/teste por viajante).
# MAGIC 4. Cria um **formulário** (widgets no topo do notebook): você preenche as preferências e recebe o **Top 3**.
# MAGIC 5. Gera o Top 3 de todos os viajantes já cadastrados e salva numa tabela Delta, que é o que o app mostra ao abrir.
# MAGIC
# MAGIC **Por que o modelo usa "interações":** uma regressão logística simples só com as colunas do perfil + destino daria o
# MAGIC **mesmo ranking para todo mundo** (o destino popular ganharia sempre). Para personalizar, o modelo aprende um peso
# MAGIC para cada combinação **preferência × destino** (ex.: "gosta de praia" × "Cancún"). Assim, perfis diferentes recebem
# MAGIC destinos diferentes. Continua sendo uma regressão logística, só com mais variáveis.
# MAGIC
# MAGIC ➡️ **Só precisa ajustar a célula de CONFIGURAÇÃO abaixo.** Depois é rodar tudo (Run all).

# COMMAND ----------

# MAGIC %md ## 1. Configuração — AJUSTE AQUI

# COMMAND ----------

# Nome da tabela Gold (813.802 avaliações, 72 colunas)
TABELA_GOLD = "workspace.default.gold_stravl_v_2"

# Nomes das colunas-chave na Gold (já preenchidos conforme o arquivo 3_Gold)
COL_VIAJANTE     = "usuario_id"
COL_DESTINO      = "destino"
COL_DESTINO_NOME = "destino"
COL_AVALIACAO    = "avaliacao"       # Sim / Não / Talvez
VALOR_SIM = "Sim"                    # <-- AJUSTE se estiver como 1, "Yes", etc.
VALOR_NAO = "Não"                    # <-- AJUSTE se estiver como 0, "No", etc.
COL_ALVO_NUM = "target"              # coluna numérica 1 = Sim / 0 = Não (tem prioridade sobre o texto acima)

# Colunas de perfil (respostas do formulário).
#   None = o notebook escolhe sozinho: todas as colunas que não são chave, não são flag de auditoria
#          e não mudam dentro do mesmo viajante (colunas do destino ficam de fora automaticamente).
#   Ou passe uma lista, ex.: ["orcamento", "estacao", "experiencias", ...]
COLUNAS_PERFIL = None

COLUNAS_EXCLUIR  = ["target", "pais"]                  # target = resposta (não pode entrar como variável); pais = do destino
PREFIXOS_EXCLUIR = ("flag_", "fl_", "ind_", "is_")     # prefixos das colunas de auditoria (ajuste ao seu padrão)
# Colunas da base original que NÃO são respostas do formulário (algoritmo usado pelo Stravl,
# recomendações dadas e avaliação delas). Qualquer coluna cujo nome contenha um destes termos sai do modelo.
TERMOS_EXCLUIR = ("model", "modelo", "retrieval", "dynamatch", "recom", "rec_", "rating", "feedback",
                  "swipe", "qtd_", "total_", "data", "timestamp")

# Se as respostas múltiplas estiverem num texto único (ex.: "Praia, Cultura"), informe o separador.
# Se já estiverem em coluna array, ou uma coluna 0/1 por opção, deixe None.
SEPARADOR_MULTIPLO = None

# Nomes amigáveis das perguntas (aparecem no formulário e na explicação da recomendação)
ROTULOS = {"form_a_pt": "Faixa etária do grupo", "form_b_pt": "Orçamento (por pessoa/noite, US$)",
           "form_c_pt": "Estação do ano", "form_f_pt": "Experiências", "form_g_pt": "Paisagens",
           "form_h_pt": "Nível de atividade", "form_i_pt": "Segurança", "form_j_pt": "Popularidade do destino",
           "form_r_pt": "Para onde quer ir", "form_rr_pt": "Regiões"}

MIN_USUARIOS_POR_RESPOSTA = 20   # respostas muito raras são ignoradas (evita ruído)
MIN_AVALIACOES_DESTINO    = 30   # só recomenda destinos com pelo menos N avaliações
TOP_K = 3

# --- Contra os "destinos campeões" (que agradam todo mundo e apareceriam para todos) ---
# PESO_POPULARIDADE: o ranking usa  prob. da pessoa − PESO × taxa de Sim geral do destino.
#   No formulário, o peso é proporcional às perguntas respondidas (todas = PESO; metade = PESO/2).
#   0 = ranking puro pela probabilidade | 1 = só "o quanto a pessoa gosta MAIS que a média"
PESO_POPULARIDADE = 0.5
# Diversidade: no máximo N destinos do mesmo país no Top 3 (None = sem limite)
COL_PAIS = "pais"          # coluna do país do destino na Gold (None se não houver)
MAX_POR_PAIS = 1
# Se a pessoa escolher regiões específicas (pergunta RR), o Top 3 mostra SÓ destinos dessas regiões
COL_REGIOES = "form_rr_pt"
FILTRAR_REGIOES = True

C_GRID = [0.005, 0.01, 0.02, 0.05]      # força da regularização testada (menor = modelo mais "conservador")
SEED = 42

# Tabela onde o Top 3 dos viajantes cadastrados será salvo
TABELA_SAIDA_TOP5 = "workspace.default.stravl_recomendacoes_top3"

# COMMAND ----------

# MAGIC %md ## 2. Leitura da Gold e escolha das variáveis

# COMMAND ----------

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

df_spark = spark.table(TABELA_GOLD)
df_spark.printSchema()

if COL_DESTINO_NOME is None:
    COL_DESTINO_NOME = COL_DESTINO
CHAVES = [COL_VIAJANTE, COL_DESTINO, COL_DESTINO_NOME, COL_AVALIACAO]

if COLUNAS_PERFIL is None:
    ignorar = set(CHAVES) | set(COLUNAS_EXCLUIR)
    candidatas = [c for c in df_spark.columns
                  if c not in ignorar
                  and not c.lower().startswith(tuple(p.lower() for p in PREFIXOS_EXCLUIR))
                  and not any(t in c.lower() for t in TERMOS_EXCLUIR)]
else:
    candidatas = list(COLUNAS_PERFIL)

extras = [c for c in [COL_PAIS, COL_ALVO_NUM] if c and c in df_spark.columns]
colunas_lidas = list(dict.fromkeys(CHAVES + candidatas + extras))
pdf = df_spark.select(*colunas_lidas).toPandas()
print(f"Linhas lidas: {len(pdf):,}")

# COMMAND ----------

def _texto(v):
    """Converte qualquer valor de célula em texto estável (para comparar e montar o formulário)."""
    if isinstance(v, (list, tuple, np.ndarray)):
        return "|".join(sorted(_texto(x) for x in v))
    if v is None:
        return ""
    if isinstance(v, float):
        if np.isnan(v):
            return ""
        if v.is_integer():
            return str(int(v))
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    return str(v).strip()

# Uma coluna de perfil não pode mudar dentro do mesmo viajante.
# Se mudar, ela descreve o DESTINO (região, país...) e não o viajante, então sai do perfil.
if COLUNAS_PERFIL is None:
    ids_amostra = pdf[COL_VIAJANTE].drop_duplicates().sample(
        min(3000, pdf[COL_VIAJANTE].nunique()), random_state=SEED)
    amostra = pdf[pdf[COL_VIAJANTE].isin(ids_amostra)]
    COLUNAS_PERFIL, removidas = [], []
    for c in candidatas:
        varia = amostra[c].map(_texto).groupby(amostra[COL_VIAJANTE]).nunique().max() > 1
        (removidas if varia else COLUNAS_PERFIL).append(c)
    print("Colunas removidas por variarem dentro do viajante (são do destino):", removidas)

print(f"\n{len(COLUNAS_PERFIL)} colunas de perfil usadas no modelo:")
for c in COLUNAS_PERFIL:
    print("  -", c)

# COMMAND ----------

# MAGIC %md ## 3. Variável-alvo e separação treino/teste **por viajante**
# MAGIC "Talvez" fica fora do treino binário (Sim = 1, Não = 0), como definido no projeto.

# COMMAND ----------

print("Valores encontrados na coluna de avaliação:")
print(pdf[COL_AVALIACAO].value_counts(dropna=False).head(10))
if COL_ALVO_NUM and COL_ALVO_NUM in pdf.columns:
    # Usa a coluna numérica (1 = Sim, 0 = Não, vazio = Talvez): não sofre com acentos
    alvo = pd.to_numeric(pdf[COL_ALVO_NUM], errors="coerce")
    dados = pdf[alvo.isin([0, 1])].copy()
    dados["y"] = alvo[alvo.isin([0, 1])].astype(np.int8).values
else:
    pdf[COL_AVALIACAO] = pdf[COL_AVALIACAO].map(_texto)
    dados = pdf[pdf[COL_AVALIACAO].isin([_texto(VALOR_SIM), _texto(VALOR_NAO)])].copy()
    dados["y"] = (dados[COL_AVALIACAO] == _texto(VALOR_SIM)).astype(np.int8)
if dados["y"].nunique() < 2:
    raise ValueError("Só uma classe encontrada (só Sim ou só Não). Confira VALOR_SIM/VALOR_NAO ou COL_ALVO_NUM.")
print(f"Avaliações Sim/Não usadas: {len(dados):,} | taxa de Sim: {dados['y'].mean():.1%}")

# Catálogo de destinos
destinos = (dados.groupby(COL_DESTINO)
            .agg(nome=(COL_DESTINO_NOME, "first"), n_aval=("y", "size"), taxa_sim=("y", "mean"))
            .reset_index())
destinos["pais"] = (dados.groupby(COL_DESTINO)[COL_PAIS].first().reindex(destinos[COL_DESTINO]).values
                    if COL_PAIS else destinos[COL_DESTINO])
# taxa de Sim geral do destino, suavizada (destino com poucas avaliações puxa para a média)
destinos["taxa_sim_suav"] = ((destinos.taxa_sim * destinos.n_aval + 20 * dados["y"].mean()) / (destinos.n_aval + 20))

# Região de cada destino (mesmas categorias da pergunta RR). Territórios dos EUA ajustados pelo nome.
REGIAO_POR_PAIS = {"Albania": "Europa",
    "Algeria": "África",
    "Andorra": "Europa",
    "Antigua and Barbuda": "Caribe",
    "Argentina": "América do Sul",
    "Armenia": "Europa",
    "Australia": "Oceania",
    "Austria": "Europa",
    "Azerbaijan": "Europa",
    "Bahamas": "Caribe",
    "Bahrain": "Oriente Médio",
    "Barbados": "Caribe",
    "Belarus": "Europa",
    "Belgium": "Europa",
    "Belize": "América do Norte",
    "Bolivia": "América do Sul",
    "Bosnia and Herzegovina": "Europa",
    "Botswana": "África",
    "Brazil": "América do Sul",
    "Brunei": "Ásia",
    "Bulgaria": "Europa",
    "Canada": "América do Norte",
    "Cape Verde": "África",
    "Chile": "América do Sul",
    "China": "Ásia",
    "Colombia": "América do Sul",
    "Costa Rica": "América do Norte",
    "Croatia": "Europa",
    "Cuba": "Caribe",
    "Cyprus": "Europa",
    "Czech Republic": "Europa",
    "Denmark": "Europa",
    "Dominica": "Caribe",
    "Dominican Republic": "Caribe",
    "Ecuador": "América do Sul",
    "Egypt": "Oriente Médio",
    "El Salvador": "América do Norte",
    "Estonia": "Europa",
    "Ethiopia": "África",
    "Fiji": "Oceania",
    "Finland": "Europa",
    "France": "Europa",
    "Georgia": "Europa",
    "Germany": "Europa",
    "Greece": "Europa",
    "Greenland (Denmark)": "América do Norte",
    "Grenada": "Caribe",
    "Guatemala": "América do Norte",
    "Guyana": "América do Sul",
    "Honduras": "América do Norte",
    "Hungary": "Europa",
    "Iceland": "Europa",
    "India": "Ásia",
    "Indonesia": "Ásia",
    "Ireland": "Europa",
    "Israel": "Oriente Médio",
    "Italy": "Europa",
    "Jamaica": "Caribe",
    "Japan": "Ásia",
    "Jordan": "Oriente Médio",
    "Kazakhstan": "Ásia",
    "Kenya": "África",
    "Kuwait": "Oriente Médio",
    "Kyrgyzstan": "Ásia",
    "Laos": "Ásia",
    "Latvia": "Europa",
    "Lebanon": "Oriente Médio",
    "Liechtenstein": "Europa",
    "Lithuania": "Europa",
    "Luxembourg": "Europa",
    "Madagascar": "África",
    "Malaysia": "Ásia",
    "Maldives": "Ásia",
    "Malta": "Europa",
    "Marshall Islands": "Oceania",
    "Mauritius": "África",
    "Mexico": "América do Norte",
    "Micronesia": "Oceania",
    "Moldova": "Europa",
    "Monaco": "Europa",
    "Montenegro": "Europa",
    "Morocco": "África",
    "Mozambique": "África",
    "Myanmar": "Ásia",
    "Namibia": "África",
    "Nauru": "Oceania",
    "Nepal": "Ásia",
    "Netherlands": "Europa",
    "New Zealand": "Oceania",
    "Nicaragua": "América do Norte",
    "Norway": "Europa",
    "Oman": "Oriente Médio",
    "Palau": "Oceania",
    "Palestine": "Oriente Médio",
    "Panama": "América do Norte",
    "Papua New Guinea": "Oceania",
    "Paraguay": "América do Sul",
    "Peru": "América do Sul",
    "Philippines": "Ásia",
    "Poland": "Europa",
    "Portugal": "Europa",
    "Qatar": "Oriente Médio",
    "Romania": "Europa",
    "Russia": "Europa",
    "Samoa": "Oceania",
    "San Marino": "Europa",
    "Sao Tome and Principe": "África",
    "Saudi Arabia": "Oriente Médio",
    "Senegal": "África",
    "Serbia": "Europa",
    "Seychelles": "África",
    "Singapore": "Ásia",
    "Slovakia": "Europa",
    "Slovenia": "Europa",
    "Solomon Islands": "Oceania",
    "South Africa": "África",
    "South Korea": "Ásia",
    "Spain": "Europa",
    "Sri Lanka": "Ásia",
    "St. Kitts and Nevis": "Caribe",
    "St. Lucia": "Caribe",
    "St. Vincent and the Grenadines": "Caribe",
    "Suriname": "América do Sul",
    "Sweden": "Europa",
    "Switzerland": "Europa",
    "Taiwan": "Ásia",
    "Tajikistan": "Ásia",
    "Tanzania": "África",
    "Thailand": "Ásia",
    "Trinidad and Tobago": "Caribe",
    "Tunisia": "África",
    "Turkey": "Europa",
    "Uganda": "África",
    "Ukraine": "Europa",
    "United Arab Emirates": "Oriente Médio",
    "United Kingdom": "Europa",
    "United States": "América do Norte",
    "Uruguay": "América do Sul",
    "Uzbekistan": "Ásia",
    "Vanuatu": "Oceania",
    "Vietnam": "Ásia",
    "Zambia": "África",
    "Zimbabwe": "África"}

def _regiao(nome, pais):
    if "Puerto Rico" in nome or "Virgin Islands" in nome: return "Caribe"
    if "Northern Mariana" in nome or "Guam" in nome: return "Oceania"
    return REGIAO_POR_PAIS.get(pais, "Outra")
destinos["regiao"] = [_regiao(str(n), p) for n, p in zip(destinos["nome"], destinos["pais"])]
print(destinos["regiao"].value_counts())
DEST_IDX = {d: i for i, d in enumerate(destinos[COL_DESTINO])}
N_DEST = len(destinos)
dados["d_idx"] = dados[COL_DESTINO].map(DEST_IDX).astype(np.int64)

# Perfis (uma linha por viajante)
perfis = dados.drop_duplicates(COL_VIAJANTE)[[COL_VIAJANTE] + COLUNAS_PERFIL].reset_index(drop=True)
USER_IDX = {u: i for i, u in enumerate(perfis[COL_VIAJANTE])}
dados["u_idx"] = dados[COL_VIAJANTE].map(USER_IDX).astype(np.int64)

# 80% dos viajantes para treino, 20% para teste (o mesmo viajante nunca aparece nos dois)
rng = np.random.default_rng(SEED)
grupo = rng.random(len(perfis))
eh_teste = grupo >= 0.8
eh_valid = (grupo >= 0.65) & (grupo < 0.8)       # parte do treino separada para escolher o C
dados["conjunto"] = np.where(eh_teste[dados["u_idx"]], "teste",
                     np.where(eh_valid[dados["u_idx"]], "validacao", "treino"))
print(dados.groupby("conjunto").agg(viajantes=(COL_VIAJANTE, "nunique"), avaliacoes=("y", "size")))

# COMMAND ----------

# MAGIC %md ## 4. Transformar o perfil em variáveis 0/1
# MAGIC Cada resposta vira uma variável. Ex.: `estacao=Verão` = 1. Respostas múltiplas acendem várias variáveis.

# COMMAND ----------

def tokens_do_valor(col, v):
    """Lista de (coluna, resposta) ativas para um valor do perfil."""
    if isinstance(v, (list, tuple, np.ndarray)):
        return [(col, _texto(x)) for x in v if _texto(x) != ""]
    s = _texto(v)
    if s == "" or s.lower() in ("nan", "none", "null"):
        return []
    if s.startswith("[") and s.endswith("]"):           # lista salva como texto: "[0, 2]" ou "['Praia', 'Cultura']"
        itens = [p.strip().strip("'\"") for p in s[1:-1].split(",")]
        return [(col, p) for p in itens if p]
    if SEPARADOR_MULTIPLO and SEPARADOR_MULTIPLO in s:
        return [(col, p.strip()) for p in s.split(SEPARADOR_MULTIPLO) if p.strip()]
    return [(col, s)]

def tokens_do_perfil(linha):
    out = []
    for col in COLUNAS_PERFIL:
        out.extend(tokens_do_valor(col, linha[col]))
    return out

# Vocabulário: respostas que aparecem em pelo menos MIN_USUARIOS_POR_RESPOSTA viajantes de treino
tokens_por_usuario = [tokens_do_perfil(r) for r in perfis[COLUNAS_PERFIL].to_dict("records")]
contagem = {}
for i, toks in enumerate(tokens_por_usuario):
    if eh_teste[i]:
        continue
    for t in set(toks):
        contagem[t] = contagem.get(t, 0) + 1
VOCAB_LISTA = sorted(t for t, n in contagem.items() if n >= MIN_USUARIOS_POR_RESPOSTA)
VOCAB = {t: j for j, t in enumerate(VOCAB_LISTA)}
N_FEAT = len(VOCAB)
print(f"Variáveis de perfil (respostas distintas): {N_FEAT}")

def matriz_de_tokens(lista_de_tokens):
    linhas, cols = [], []
    for i, toks in enumerate(lista_de_tokens):
        for t in set(toks):
            j = VOCAB.get(t)
            if j is not None:
                linhas.append(i); cols.append(j)
    return sp.csr_matrix((np.ones(len(linhas), dtype=np.float32), (linhas, cols)),
                         shape=(len(lista_de_tokens), N_FEAT))

U = matriz_de_tokens(tokens_por_usuario)   # viajantes × respostas

def montar_X(u_idx, d_idx):
    """Matriz do modelo: [destino] + [perfil] + [perfil × destino]."""
    Ur = U[u_idx]
    n = Ur.shape[0]
    bloco_dest = sp.csr_matrix((np.ones(n, dtype=np.float32), (np.arange(n), d_idx)), shape=(n, N_DEST))
    nnz = np.diff(Ur.indptr)
    idx_inter = np.repeat(d_idx, nnz) * N_FEAT + Ur.indices
    bloco_inter = sp.csr_matrix((Ur.data, idx_inter, Ur.indptr), shape=(n, N_DEST * N_FEAT))
    return sp.hstack([bloco_dest, Ur, bloco_inter], format="csr")

# COMMAND ----------

# MAGIC %md ## 5. Treino da Regressão Logística (escolhendo a regularização na validação)

# COMMAND ----------

def subconjunto(nome):
    m = dados["conjunto"].isin(nome if isinstance(nome, list) else [nome])
    return dados.loc[m].reset_index(drop=True)

tr, va, te = subconjunto("treino"), subconjunto("validacao"), subconjunto("teste")
X_tr = montar_X(tr["u_idx"].values, tr["d_idx"].values)
X_va = montar_X(va["u_idx"].values, va["d_idx"].values)

resultados_c = {}
for C in C_GRID:
    m = LogisticRegression(C=C, solver="liblinear", max_iter=300)
    m.fit(X_tr, tr["y"].values)
    resultados_c[C] = roc_auc_score(va["y"].values, m.decision_function(X_va))
    print(f"C={C:<6} AUC validação = {resultados_c[C]:.4f}")
MELHOR_C = max(resultados_c, key=resultados_c.get)
print("Melhor C:", MELHOR_C)

# Treina de novo com treino + validação
trva = subconjunto(["treino", "validacao"])
modelo = LogisticRegression(C=MELHOR_C, solver="liblinear", max_iter=300)
modelo.fit(montar_X(trva["u_idx"].values, trva["d_idx"].values), trva["y"].values)

# COMMAND ----------

# MAGIC %md ## 6. Avaliação em viajantes nunca vistos
# MAGIC - **AUC:** quanto o modelo separa Sim de Não (0,5 = sorteio; 1,0 = perfeito).
# MAGIC - **Precisão@3:** dos 5 destinos que o modelo colocaria no topo (entre os que o viajante avaliou), quantos ele disse Sim.
# MAGIC - **Acerto@3:** % de viajantes com pelo menos 1 "Sim" no Top 3.
# MAGIC - Comparamos com **Popularidade** (todo mundo recebe os destinos com maior taxa de Sim) e **Aleatório**.

# COMMAND ----------

def metricas(df_rows, scores, minimo=10):
    t = df_rows[[COL_VIAJANTE, "y"]].assign(s=scores)
    t = t.sort_values([COL_VIAJANTE, "s"], ascending=[True, False])
    n_por_user = t.groupby(COL_VIAJANTE)["y"].size()
    validos = n_por_user[n_por_user >= minimo].index
    t = t[t[COL_VIAJANTE].isin(validos)]
    top = t.groupby(COL_VIAJANTE).head(TOP_K)
    return {
        "AUC": roc_auc_score(df_rows["y"], scores),
        f"Precisão@{TOP_K}": top.groupby(COL_VIAJANTE)["y"].mean().mean(),
        f"Acerto@{TOP_K}": top.groupby(COL_VIAJANTE)["y"].max().mean(),
    }

# Popularidade (taxa de Sim suavizada, calculada só com o treino)
pop = trva.groupby("d_idx")["y"].agg(["sum", "size"]).reindex(range(N_DEST), fill_value=0)
taxa_global = trva["y"].mean()
POP_SCORE = ((pop["sum"] + 20 * taxa_global) / (pop["size"] + 20)).values

s_modelo = modelo.decision_function(montar_X(te["u_idx"].values, te["d_idx"].values))
s_pop = POP_SCORE[te["d_idx"].values]
s_rand = np.random.default_rng(SEED).random(len(te))

tabela_metricas = pd.DataFrame({
    "Regressão Logística (personalizada)": metricas(te, s_modelo),
    "Popularidade": metricas(te, s_pop),
    "Aleatório": metricas(te, s_rand),
}).T.round(4)
display(tabela_metricas)

# COMMAND ----------

# MAGIC %md ## 7. Modelo final (todos os viajantes) e função de recomendação

# COMMAND ----------

modelo_final = LogisticRegression(C=MELHOR_C, solver="liblinear", max_iter=300)
modelo_final.fit(montar_X(dados["u_idx"].values, dados["d_idx"].values), dados["y"].values)

def pesos_do_modelo(m):
    coef = m.coef_.ravel()
    return {"B0": m.intercept_[0], "W_DEST": coef[:N_DEST], "W_PERFIL": coef[N_DEST:N_DEST + N_FEAT],
            # peso de cada resposta para cada destino
            "W_INTER": coef[N_DEST + N_FEAT:].reshape(N_DEST, N_FEAT)}

PESOS = pesos_do_modelo(modelo_final)
B0, W_DEST, W_PERFIL, W_INTER = PESOS["B0"], PESOS["W_DEST"], PESOS["W_PERFIL"], PESOS["W_INTER"]
ELEGIVEL = destinos["n_aval"].values >= MIN_AVALIACOES_DESTINO
POP = destinos["taxa_sim_suav"].values
PAIS_IDX = pd.factorize(destinos["pais"])[0]

def probabilidades(Um, pesos=None):
    """Probabilidade de 'Sim' para TODOS os destinos. Um = matriz (viajantes × respostas)."""
    w = pesos or PESOS
    logit = (w["B0"] + w["W_DEST"][None, :] + np.asarray(Um @ w["W_PERFIL"]).reshape(-1, 1)
             + np.asarray(Um @ w["W_INTER"].T))
    return 1 / (1 + np.exp(-logit))

REGIAO_DEST = destinos["regiao"].values

def regioes_do_perfil(valor):
    """Lista de regiões escolhidas na pergunta RR (vazia = qualquer lugar)."""
    if not FILTRAR_REGIOES or valor is None:
        return []
    return [r for _, r in tokens_do_valor(COL_REGIOES, valor)]

def escolher_top(p, excluir=None, k=None, peso=None, max_pais=None, regioes=None):
    """Escolhe os k destinos: (prob. da pessoa − peso × taxa geral), só elegíveis, limite por país,
    e só das regiões escolhidas (se houver)."""
    k = TOP_K if k is None else k
    peso = PESO_POPULARIDADE if peso is None else peso
    max_pais = MAX_POR_PAIS if max_pais is None else max_pais
    score = np.where(ELEGIVEL, p - peso * POP, -np.inf)
    if regioes:
        score = np.where(np.isin(REGIAO_DEST, list(regioes)), score, -np.inf)
    if excluir is not None and len(excluir):
        score[np.asarray(excluir)] = -np.inf
    escolhidos, por_pais = [], {}
    for d in np.argsort(-score):
        if not np.isfinite(score[d]) or len(escolhidos) == k:
            break
        pa = PAIS_IDX[d]
        if max_pais and por_pais.get(pa, 0) >= max_pais:
            continue
        por_pais[pa] = por_pais.get(pa, 0) + 1
        escolhidos.append(d)
    return escolhidos

def _rotulo(t):
    return f"{ROTULOS.get(t[0], t[0])}: {t[1]}"

def recomendar(perfil, k=None, excluir=None, peso=None, max_pais=None):
    """
    perfil: dicionário {coluna: valor ou lista de valores}, com as mesmas respostas do formulário.
    Retorna os k destinos recomendados + as preferências que mais pesaram.
    """
    toks = []
    for col, v in perfil.items():
        toks.extend(tokens_do_valor(col, list(v) if isinstance(v, (list, tuple, set)) else v))
    Um = matriz_de_tokens([toks])
    if peso is None:
        # Ajuste contra campeões proporcional ao quanto conhecemos a pessoa:
        # com poucas respostas, o modelo sabe pouco e é melhor confiar mais na popularidade.
        principais = [c for c in COLUNAS_PERFIL if c not in (COL_REGIOES, "form_r_pt")]
        respondidas = sum(1 for c in principais if len(tokens_do_valor(c, perfil.get(c))) > 0)
        peso = PESO_POPULARIDADE * respondidas / max(len(principais), 1)
    p = probabilidades(Um)[0]
    ativos = Um.indices
    linhas = []
    regs = regioes_do_perfil(perfil.get(COL_REGIOES))
    for pos, d in enumerate(escolher_top(p, excluir, k, peso, max_pais, regs), 1):
        pesos = W_INTER[d, ativos]
        ordem = [ativos[i] for i in np.argsort(-pesos)[:3] if pesos[i] > 0]
        linhas.append({
            "posição": pos,
            "destino": destinos.loc[d, "nome"],
            "região": destinos.loc[d, "regiao"],
            "probabilidade_sim": round(float(p[d]), 3),
            "vs. média do destino": f"{(p[d] - POP[d]) * 100:+.0f} p.p.",
            "por que (preferências que mais pesaram)": "; ".join(_rotulo(VOCAB_LISTA[j]) for j in ordem),
        })
    return pd.DataFrame(linhas)

# Teste: dois viajantes reais com perfis diferentes devem receber listas diferentes
for i in perfis.sample(2, random_state=SEED).index:
    print("Viajante", perfis.loc[i, COL_VIAJANTE])
    display(recomendar({c: perfis.loc[i, c] for c in COLUNAS_PERFIL}))

# COMMAND ----------

# MAGIC %md ### 7.1 Ajuste contra os "destinos campeões": acerto × variedade
# MAGIC Avaliado só nos viajantes de teste, com o modelo que não os viu. Para cada peso:
# MAGIC - **Precisão@3**: entre os destinos que o viajante avaliou, quantos do Top 3 ele aprovou (mede acerto).
# MAGIC - **Destinos distintos** nos Top 3 do catálogo inteiro e **% das vagas ocupadas pelos 10 mais frequentes** (medem variedade).

# COMMAND ----------

PESOS_TESTE = pesos_do_modelo(modelo)
te_users = np.unique(te["u_idx"].values)
P_te = probabilidades(U[te_users], PESOS_TESTE)
linha_de = {u: i for i, u in enumerate(te_users)}
lin = np.array([linha_de[u] for u in te["u_idx"].values])
p_linha = P_te[lin, te["d_idx"].values]
comparacao = []
for peso in [0, 0.25, 0.5, 0.75, 1.0]:
    for mp in [None, 1]:
        s = p_linha - peso * POP[te["d_idx"].values]
        m = metricas(te, s)
        tops = [escolher_top(P_te[i], peso=peso, max_pais=mp or 0) for i in range(len(te_users))]
        cont = pd.Series(np.concatenate(tops)).value_counts()
        comparacao.append({"peso_popularidade": peso, "máx. por país": mp or "sem limite",
                           f"Precisão@{TOP_K}": round(m[f"Precisão@{TOP_K}"], 4),
                           f"destinos distintos nos Top {TOP_K}": len(cont),
                           "% vagas dos 10 mais frequentes": round(cont.head(10).sum() / cont.sum(), 3)})
display(pd.DataFrame(comparacao))
print(f"Configuração em uso: PESO_POPULARIDADE={PESO_POPULARIDADE}, MAX_POR_PAIS={MAX_POR_PAIS}")

# COMMAND ----------

# MAGIC %md ## 8. Formulário 📝
# MAGIC Rodar esta célula cria os campos **no topo do notebook**. Preencha e depois rode a célula seguinte.
# MAGIC (Pode marcar mais de uma opção. Deixe "(não informado)" para pular.)

# COMMAND ----------

NAO_INF = "(não informado)"
opcoes = {}
for col, val in VOCAB_LISTA:
    opcoes.setdefault(col, []).append(val)

dbutils.widgets.removeAll()
MAPA_WIDGETS = {}    # widget -> {texto exibido: valor real}
for col in COLUNAS_PERFIL:
    if col not in opcoes:
        continue
    mapa = {v.replace(",", " ·"): v for v in sorted(opcoes[col])}   # vírgula quebra o multiselect
    MAPA_WIDGETS[col] = mapa
    dbutils.widgets.multiselect(col, NAO_INF, [NAO_INF] + list(mapa.keys()), ROTULOS.get(col, col))
print(f"{len(MAPA_WIDGETS)} campos criados no topo do notebook.")

# COMMAND ----------

# MAGIC %md ## 9. Resultado do formulário — Top 3 do usuário

# COMMAND ----------

perfil_form = {}
for col, mapa in MAPA_WIDGETS.items():
    marcados = [x for x in dbutils.widgets.get(col).split(",") if x and x != NAO_INF]
    if marcados:
        perfil_form[col] = [mapa[x] for x in marcados if x in mapa]

print("Preferências informadas:", perfil_form if perfil_form else "nenhuma (resultado = destinos mais populares)")
display(recomendar(perfil_form))

# COMMAND ----------

# MAGIC %md ## 10. Top 3 de todos os viajantes cadastrados → tabela Delta
# MAGIC É o que o app mostra quando um usuário já cadastrado abre a tela inicial.
# MAGIC Destinos que o viajante já marcou como "Não" ficam de fora.

# COMMAND ----------

nao_por_user = dados[dados["y"] == 0].groupby("u_idx")["d_idx"].apply(np.array).to_dict()
saida = []
LOTE = 4000
for ini in range(0, U.shape[0], LOTE):
    fim = min(ini + LOTE, U.shape[0])
    P = probabilidades(U[ini:fim])
    for i in range(fim - ini):
        u = ini + i
        regs = regioes_do_perfil(perfis.loc[u, COL_REGIOES]) if COL_REGIOES in perfis.columns else []
        for pos, d in enumerate(escolher_top(P[i], nao_por_user.get(u), regioes=regs), 1):
            saida.append((perfis.loc[u, COL_VIAJANTE], pos, destinos.loc[d, COL_DESTINO],
                          destinos.loc[d, "nome"], float(P[i, d])))

top5 = pd.DataFrame(saida, columns=["viajante", "posicao", "codigo_destino", "destino", "probabilidade_sim"])
if COL_DESTINO == COL_DESTINO_NOME:
    top5 = top5.drop(columns="codigo_destino")
top5 = top5.astype({"viajante": str, "destino": str})

print(f"Destinos diferentes aparecendo nos Top {TOP_K}: {top5['destino'].nunique()} "
      f"(na recomendação por popularidade seriam só {TOP_K})")
spark.createDataFrame(top5).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(TABELA_SAIDA_TOP5)
print("Salvo em", TABELA_SAIDA_TOP5)
display(top5.head(20))

# COMMAND ----------

# MAGIC %md ## 11. Registro no MLflow (parâmetros, métricas e modelo)

# COMMAND ----------

import joblib, tempfile, os
try:
    import mlflow
    with mlflow.start_run(run_name="stravl_regressao_logistica"):
        mlflow.log_params({"C": MELHOR_C, "n_variaveis_perfil": N_FEAT, "n_destinos": N_DEST,
                           "min_aval_destino": MIN_AVALIACOES_DESTINO})
        for nome, val in tabela_metricas.loc["Regressão Logística (personalizada)"].items():
            mlflow.log_metric(nome.replace("@", "_at_").replace("ã", "a"), float(val))
        caminho = os.path.join(tempfile.mkdtemp(), "modelo_stravl.joblib")
        joblib.dump({"B0": B0, "W_DEST": W_DEST, "W_PERFIL": W_PERFIL, "W_INTER": W_INTER,
                     "VOCAB_LISTA": VOCAB_LISTA, "COLUNAS_PERFIL": COLUNAS_PERFIL,
                     "destinos": destinos, "ELEGIVEL": ELEGIVEL}, caminho)
        mlflow.log_artifact(caminho)
    print("Registrado no MLflow (aba Experiments).")
except Exception as e:
    print("MLflow não registrado:", e)
