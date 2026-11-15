# Compass

Sistema de recomendação de destinos de viagem desenvolvido como trabalho final da disciplina Hands-On Engenharia de Dados em Big Data e Analytics, do MBA em Engenharia de Dados da Universidade Presbiteriana Mackenzie.

A partir de um formulário de preferências (faixa etária, orçamento, período da viagem, experiências, paisagens, nível de atividade, segurança, popularidade e regiões), o sistema recomenda os 3 destinos mais compatíveis com o perfil do viajante.

- Aplicação: https://renatodeoliveira101520.github.io/recomendador-viagens/
- Briefing do projeto: [COMPASS.pdf](COMPASS.pdf)
- Arquitetura de CI/CD: [docs/ARQUITETURA_CICD.md](docs/ARQUITETURA_CICD.md)

## Contexto

Plataformas de viagem costumam recomendar os mesmos destinos populares para todos os usuários. Na base analisada, 61% das avaliações foram negativas e 22% dos viajantes recusaram todos os destinos apresentados. O objetivo do projeto é construir um pipeline de dados e um modelo que gerem recomendações personalizadas e diversas.

## Base de dados

Stravl Travel Preference Data ([github.com/Stravl/stravl-data](https://github.com/Stravl/stravl-data)), base pública liberada para uso acadêmico e comercial. Os dados são baixados diretamente da fonte pelo notebook de ingestão.

| Item | Quantidade |
|---|---:|
| Viajantes no formulário | 80.301 |
| Viajantes com avaliações | 61.130 |
| Destinos avaliados | 1.452 |
| Avaliações (Sim, Não, Talvez) | 813.802 |

## Arquitetura

O pipeline roda no Databricks, com tabelas Delta organizadas em arquitetura medalhão no schema `workspace.default`.

```
Stravl (GitHub)
   │
   ▼
Bronze   dados originais, sem transformação
   │
   ▼
Silver   respostas traduzidas, flags de qualidade, país e região dos destinos
   │
   ▼
Gold     uma linha por avaliação (viajante × destino) com o perfil do viajante
   │
   ├──► EDA
   └──► Modelo ──► Top 3 por viajante (Delta) ──► Aplicação (GitHub Pages)
```

| Notebook | Tabelas geradas | Descrição |
|---|---|---|
| `01_bronze.py` | `bronze_stravl_formulario`, `bronze_stravl_destinos` | Ingestão dos arquivos originais |
| `02_silver.py` | `silver_stravl_viajantes`, `silver_stravl_destinos` | Conversão das respostas em listas, tradução dos códigos, flags de qualidade e enriquecimento dos destinos |
| `03_gold.py` | `gold_stravl_v_2` | Consolidação das avaliações e junção com o perfil |
| `04_eda.py` | — | Análise exploratória |
| `05_modelo_recomendacao.py` | `stravl_recomendacoes_top3` | Treino, avaliação e geração do Top 3 |

### Tratamento dos dados

- Registros com problemas de qualidade são sinalizados por flags, e não removidos. Exemplos: todas as faixas de orçamento marcadas, formulário incompleto e regiões inconsistentes com a resposta "para onde quer ir".
- Avaliações repetidas do mesmo viajante para o mesmo destino são contadas uma única vez.
- Pares com respostas conflitantes (por exemplo, Sim e Não para o mesmo destino) são descartados.

### Tabela Gold

`gold_stravl_v_2`: 813.802 linhas e 15 colunas.

| Coluna | Descrição |
|---|---|
| `usuario_id` | Identificador do viajante |
| `destino`, `pais` | Destino avaliado e país |
| `form_a_pt` a `form_rr_pt` | Respostas do formulário, traduzidas (listas em formato JSON) |
| `avaliacao` | Sim, Não ou Talvez |
| `target` | 1 = Sim, 0 = Não, nulo = Talvez |

## Análise exploratória

- 22% dos viajantes recusaram todos os destinos avaliados; 4% aprovaram todos.
- 21% marcaram todas as faixas de orçamento e idade, o que indica que pularam essas perguntas.
- Destinos nos EUA concentram 40% das avaliações e têm a menor taxa de aprovação (21%, contra 47% no restante).
- As preferências influenciam pouco o volume de aprovações, mas mudam quais destinos são aprovados. Quem busca praia aprova mais Belize e Bahamas; quem busca montanha, Füssen e Pokhara.
- A segmentação com K-Means identificou cinco perfis de viajante.

## Modelo

Regressão logística que estima a probabilidade de o viajante aprovar cada destino.

- Alvo: Sim = 1 e Não = 0. As respostas "Talvez" não entram no treino.
- Variáveis: respostas do formulário em formato binário, destino e interações entre preferência e destino. Sem as interações, o ranking seria igual para todos os usuários.
- Regularização L2, com o parâmetro `C` escolhido pela AUC na validação.
- Treino, validação e teste separados por viajante.
- Métricas e artefatos registrados no MLflow.

Após o modelo, três regras são aplicadas ao ranking:

1. Penalização de destinos muito populares: `score = probabilidade − 0,5 × taxa de aprovação geral do destino`.
2. No máximo um destino por país no Top 3.
3. Quando o usuário escolhe regiões específicas, apenas destinos dessas regiões são recomendados.

### Resultados

Avaliação nos viajantes do conjunto de teste:

| Métrica | Modelo | Popularidade | Aleatório |
|---|---:|---:|---:|
| Precisão@3 | 52,7% | 52,6% | 36,2% |
| Acerto@3 (ao menos 1 aprovação no Top 3) | 70,0% | 69,3% | 60,4% |
| AUC | 0,737 | 0,708 | 0,499 |
| Destinos distintos recomendados | 504 | 3 | — |

O modelo tem precisão equivalente à recomendação por popularidade, mas recomenda 504 destinos diferentes, contra 3. O Top 3 foi escolhido por equilibrar precisão e taxa de acerto: no Top 5, o acerto sobe 3 pontos e a precisão cai 4.

As métricas consideram apenas os destinos que cada viajante avaliou (cerca de 10 por pessoa), o que torna a estimativa conservadora.

## Como executar

1. No Databricks, crie uma Git folder com a URL deste repositório (*Workspace > Create > Git folder*).
2. Execute os notebooks na ordem, de `01` a `05`, com computação Serverless.
3. Caso o workspace não tenha acesso à internet, envie `Stravl_Travel_Preference_Data.csv` e `destination_ids.txt` para o Volume `/Volumes/workspace/default/stravl_raw`.
4. Para testar uma recomendação, abra o `05_modelo_recomendacao`, preencha os parâmetros do formulário e execute a seção 9.

O catálogo e o schema podem ser alterados nas variáveis `CATALOGO` e `SCHEMA`, no início de cada notebook.

## CI/CD

A proposta de integração e entrega contínua está descrita em [docs/ARQUITETURA_CICD.md](docs/ARQUITETURA_CICD.md). Ela inclui testes com GitHub Actions, deploy com Databricks Asset Bundles, infraestrutura com Terraform e promoção do modelo condicionada à Precisão@3 ser igual ou superior à da recomendação por popularidade.

## Limitações

- A maioria dos viajantes avaliou apenas 10 destinos.
- A pergunta sobre regiões foi respondida por 21% dos viajantes.
- A estação do ano segue o calendário do hemisfério norte, já que a maioria dos respondentes é dos EUA. Por isso a aplicação exibe o período do ano.
- O orçamento está em dólares, por pessoa e por noite.
- A base não contém atributos dos destinos, como clima e custo.

## Próximos passos

- Filtragem colaborativa a partir das avaliações.
- Inclusão de atributos dos destinos.
- Uso do feedback da aplicação para retreinar o modelo.
- Implementação do pipeline de CI/CD.

## Equipe

- Renato de Oliveira de Souza
- Luciano Gomes da Silva
- José Carlos Marques
