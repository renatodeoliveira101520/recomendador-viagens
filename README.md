# Compass

Sistema de recomendação de destinos de viagem desenvolvido como trabalho final da disciplina Hands-On Engenharia de Dados em Big Data e Analytics, do MBA em Engenharia de Dados da Universidade Presbiteriana Mackenzie.

A partir de um formulário de preferências (faixa etária, orçamento, período da viagem, experiências, paisagens, nível de atividade, segurança, popularidade e regiões), o sistema recomenda os 3 destinos mais compatíveis com o perfil do viajante, com o grau de compatibilidade e o motivo de cada recomendação.

- Aplicação: https://renatodeoliveira101520.github.io/recomendador-viagens/
- Análise exploratória (notebook renderizado): [stravl_eda.html](https://renatodeoliveira101520.github.io/recomendador-viagens/stravl_eda.html)
- Modelo (notebook renderizado): [stravl_modelo_recomendacao.html](https://renatodeoliveira101520.github.io/recomendador-viagens/stravl_modelo_recomendacao.html)
- Briefing do projeto: [COMPASS.pdf](COMPASS.pdf)
- Arquitetura de CI/CD: [docs/ARQUITETURA_CICD.md](docs/ARQUITETURA_CICD.md)

## Contexto

Plataformas de viagem costumam recomendar os mesmos destinos populares para todos os usuários, sem considerar o perfil de cada viajante. Na base analisada, 61% das avaliações foram negativas e 22% dos viajantes recusaram todos os destinos apresentados. Um ranking apenas por popularidade entregaria os mesmos 3 destinos para todos, deixando sem visibilidade a maior parte de um catálogo de 1.452 destinos.

O objetivo do projeto é construir, de ponta a ponta, um pipeline de dados no Databricks e um modelo de machine learning que gerem recomendações personalizadas, explicáveis e diversas.

## Base de dados

Stravl Travel Preference Data ([github.com/Stravl/stravl-data](https://github.com/Stravl/stravl-data)), base pública liberada para uso acadêmico e comercial. Os dados não são copiados para este repositório: o notebook de ingestão faz o download diretamente da fonte.

| Item | Quantidade |
|---|---:|
| Viajantes no formulário | 80.301 |
| Viajantes com avaliações | 61.130 |
| Destinos avaliados | 1.452 |
| Avaliações (Sim, Não, Talvez) | 813.802 |
| Taxa de aprovação (Sim ÷ (Sim + Não)) | 36,7% |

Cada viajante respondeu a um formulário de preferências e depois avaliou destinos com Sim, Não ou Talvez.

Arquivos de origem:

- `Stravl_Travel_Preference_Data.csv`: respostas do formulário e avaliações de cada viajante.
- `destination_ids.txt`: catálogo com o código e o nome de cada destino.

## Arquitetura

O pipeline roda no Databricks (computação Serverless e Unity Catalog), com tabelas Delta organizadas em arquitetura medalhão no schema `workspace.default`. Todas as camadas são regravadas por completo a cada execução, o que torna o pipeline idempotente: executar de novo gera sempre o mesmo resultado.

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
                └─► MLflow (parâmetros, métricas e artefatos)
```

| Notebook | Tabelas geradas | Descrição |
|---|---|---|
| `01_bronze.py` | `bronze_stravl_formulario`, `bronze_stravl_destinos` | Ingestão dos arquivos originais, com data e origem de cada carga |
| `02_silver.py` | `silver_stravl_viajantes`, `silver_stravl_destinos` | Conversão das respostas em listas, tradução dos códigos, flags de qualidade e enriquecimento dos destinos |
| `03_gold.py` | `gold_stravl_v_2` | Consolidação das avaliações e junção com o perfil |
| `04_eda.py` | — | Análise exploratória e segmentação de perfis |
| `05_modelo_recomendacao.py` | `stravl_recomendacoes_top3` | Treino, avaliação, formulário de teste e geração do Top 3 |

## Tratamento dos dados

### Silver

As respostas do formulário chegam como texto (por exemplo, `"['0', '1']"`) e são convertidas em listas. Os códigos são traduzidos para português conforme o dicionário de dados do Stravl:

| Pergunta | Opções |
|---|---|
| A. Faixa etária do grupo | 0-19, 20-39, 40-59, 60+ anos |
| B. Orçamento (US$ por pessoa por noite) | 0-49, 50-99, 100-249, 300+ |
| C. Estação da viagem | Inverno, Primavera, Verão, Outono |
| F. Experiências | Praia, Aventura, Natureza, Cultura, Vida noturna, História, Compras, Gastronomia |
| G. Paisagens | Urbana, Rural, Mar, Montanha, Lago, Deserto, Planícies, Selva |
| H. Nível de atividade | Relaxado e tranquilo, Equilibrado, Ativo |
| I. Segurança | Muito preocupado, Equilibrado, Preparado para tudo |
| J. Popularidade | Fora dos caminhos comuns, Local clássico, Popular e moderno |
| R. Para onde quer ir | Em qualquer lugar, Regiões específicas |
| RR. Regiões | Europa, América do Norte, Caribe, Ásia, América do Sul, Oriente Médio, África, Oceania |

Os destinos recebem país (extraído do nome) e região (mesmas categorias da pergunta RR).

### Qualidade de dados

Registros com problemas de qualidade são sinalizados, e não removidos. Assim, nenhuma informação se perde e cada análise decide como tratá-los.

| Flag | Regra |
|---|---|
| `flag_orcamento_todos` | Marcou todas as faixas de orçamento |
| `flag_idade_todas` | Marcou todas as faixas etárias |
| `flag_popularidade_todas` | Marcou todas as opções de popularidade |
| `flag_geo_inconsistente` | "Em qualquer lugar" com regiões marcadas, ou "Regiões específicas" sem nenhuma região |
| `flag_formulario_incompleto` | Deixou em branco alguma das perguntas A, B, C, F ou G |
| `flag_sem_avaliacoes` | Não avaliou nenhum destino |

### Gold

- Avaliações repetidas do mesmo viajante para o mesmo destino são contadas uma única vez.
- Pares com respostas conflitantes (por exemplo, Sim e Não para o mesmo destino) são descartados.
- Ao final, o notebook confere os totais esperados: 813.802 avaliações, 61.130 viajantes e 1.452 destinos.

`gold_stravl_v_2`: 813.802 linhas e 15 colunas.

| Coluna | Descrição |
|---|---|
| `usuario_id` | Identificador do viajante |
| `destino`, `pais` | Destino avaliado e país |
| `form_a_pt` a `form_rr_pt` | Respostas do formulário traduzidas (listas em formato JSON) |
| `avaliacao` | Sim, Não ou Talvez |
| `target` | 1 = Sim, 0 = Não, nulo = Talvez |

## Análise exploratória

- 22% dos viajantes recusaram todos os destinos avaliados; 4% aprovaram todos.
- 21% marcaram todas as faixas de orçamento e idade, o que indica que pularam essas perguntas.
- Destinos nos EUA concentram 40% das avaliações e têm a menor taxa de aprovação (21%, contra 47% no restante).
- As preferências influenciam pouco o volume de aprovações, mas mudam quais destinos são aprovados. Quem busca praia aprova mais Belize e Bahamas; quem busca montanha, Füssen e Pokhara; quem busca deserto, destinos no Egito.
- A segmentação com K-Means identificou cinco perfis de viajante, como natureza e montanha, praia e mar, e cultura urbana.

Esses achados orientaram o desenho do modelo: como o perfil muda quais destinos agradam, o modelo precisa aprender a relação entre cada preferência e cada destino.

## Modelo

Regressão logística que estima a probabilidade de o viajante aprovar cada destino.

| Item | Definição |
|---|---|
| Alvo | Sim = 1 e Não = 0. As respostas "Talvez" não entram no treino |
| Variáveis | Respostas do formulário em formato binário, destino e interações entre preferência e destino |
| Regularização | L2, com o parâmetro `C` escolhido pela AUC na validação |
| Filtros | Respostas com menos de 20 viajantes são ignoradas; só são recomendados destinos com pelo menos 30 avaliações |
| Validação | Treino, validação e teste separados por viajante: o modelo é avaliado em pessoas que não viu no treino |
| Rastreamento | Parâmetros, métricas e pesos registrados no MLflow |

Sem as interações entre preferência e destino, o ranking seria igual para todos os usuários, e os destinos mais populares apareceriam sempre no topo.

### Regras de negócio

Após o modelo, três regras são aplicadas ao ranking:

1. Penalização de destinos muito populares: `score = probabilidade − 0,5 × taxa de aprovação geral do destino`. No formulário, a penalização é proporcional ao número de perguntas respondidas.
2. No máximo um destino por país no Top 3.
3. Quando o usuário escolhe regiões específicas, apenas destinos dessas regiões são recomendados.

## Resultados

Avaliação nos viajantes do conjunto de teste:

| Métrica | Modelo | Popularidade | Aleatório |
|---|---:|---:|---:|
| Precisão@3 (aprovados entre os 3 recomendados) | 52,7% | 52,6% | 36,2% |
| Acerto@3 (ao menos 1 aprovação no Top 3) | 70,0% | 69,3% | 60,4% |
| Destinos distintos recomendados | 504 | 3 | — |
| AUC (probabilidade do modelo, antes das regras de negócio) | 0,737 | 0,708 | 0,499 |
| Acurácia | 69,6% | — | 63,4%¹ |

¹ Acurácia de prever "Não" para todos os casos.

O modelo tem precisão equivalente à recomendação por popularidade, mas recomenda 504 destinos diferentes, contra 3. O ganho do projeto está na personalização e na diversidade, e não em um aumento de precisão.

### Modelo escolhido

A regressão logística com interações entre preferência e destino foi escolhida como modelo final pelos seguintes motivos:

- Supera as duas referências em AUC e em Acerto@3, com Precisão@3 equivalente à da popularidade.
- É a única abordagem que personaliza a recomendação: perfis diferentes recebem destinos diferentes, enquanto a popularidade entrega os mesmos 3 destinos para todos.
- É interpretável. Os pesos do modelo mostram quais preferências mais influenciaram cada recomendação, e essa explicação aparece para o usuário na aplicação.
- É leve. Os pesos são exportados e o cálculo roda direto no navegador, sem necessidade de servidor.

### Tamanho da lista

| | Top 1 | Top 3 | Top 5 |
|---|---:|---:|---:|
| Precisão | 57,2% | 52,7% | 48,6% |
| Acerto (ao menos 1 aprovação) | 57,2% | 70,0% | 73,1% |

O Top 3 foi escolhido por equilibrar precisão e taxa de acerto: do Top 3 para o Top 5, o acerto sobe 3 pontos e a precisão cai 4.

### Efeito das regras de diversidade

| | Sem regras | Com regras |
|---|---:|---:|
| Precisão@3 | 53,9% | 52,7% |
| Destinos distintos recomendados | 189 | 504 |

As regras reduzem a precisão em 1,2 ponto e aumentam em 2,7 vezes a variedade de destinos recomendados.

As métricas consideram apenas os destinos que cada viajante avaliou (cerca de 10 por pessoa), e não o catálogo inteiro, o que torna a estimativa conservadora.

## Aplicação

A aplicação está publicada no GitHub Pages (`index.html`) e roda no navegador com os pesos exportados do modelo.

- À esquerda, o formulário de perfil.
- À direita, o Top 3, com foto, região, grau de compatibilidade (Match) e as preferências que mais pesaram em cada recomendação.
- O botão "Ver outras 3 sugestões" mostra os próximos destinos do ranking.
- O quadro "Ver os inputs enviados ao modelo" exibe exatamente o que foi enviado ao modelo.

O Match indica o quanto cada destino combina com o perfil em comparação com os demais destinos disponíveis e segue a ordem do ranking.

## Como executar

Pré-requisitos: workspace Databricks com Unity Catalog, computação Serverless e permissão para criar tabelas em `workspace.default`.

1. No Databricks, crie uma Git folder com a URL deste repositório (*Workspace > Create > Git folder*).
2. Execute os notebooks na ordem, de `01` a `05`.
3. Caso o workspace não tenha acesso à internet, envie `Stravl_Travel_Preference_Data.csv` e `destination_ids.txt` para o Volume `/Volumes/workspace/default/stravl_raw`.
4. Confira os totais exibidos ao final do `03_gold` (813.802 avaliações, 61.130 viajantes e 1.452 destinos).
5. Para testar uma recomendação, abra o `05_modelo_recomendacao`, preencha os parâmetros do formulário e execute a seção 9.

O catálogo e o schema podem ser alterados nas variáveis `CATALOGO` e `SCHEMA`, no início de cada notebook.

## Estrutura do repositório

```
recomendador-viagens/
├── 01_bronze.py
├── 02_silver.py
├── 03_gold.py
├── 04_eda.py
├── 05_modelo_recomendacao.py
├── stravl_eda.html                  EDA exportada com resultados
├── stravl_modelo_recomendacao.html  Modelo exportado com resultados
├── index.html                       Aplicação Compass
├── COMPASS.pdf                      Briefing do projeto
├── docs/
│   └── ARQUITETURA_CICD.md          Proposta de CI/CD e infraestrutura
└── README.md
```

## Tecnologias

| Finalidade | Tecnologia |
|---|---|
| Plataforma | Databricks (Serverless, Unity Catalog) |
| Armazenamento | Delta Lake |
| Processamento | PySpark |
| Análise e modelagem | pandas, NumPy, SciPy, scikit-learn |
| Rastreamento de experimentos | MLflow |
| Versionamento | Git e GitHub (Databricks Git folders) |
| Aplicação | HTML e JavaScript, GitHub Pages |
| CI/CD e infraestrutura (proposta) | GitHub Actions, Databricks Asset Bundles, Terraform |

## CI/CD

A proposta de integração e entrega contínua está descrita em [docs/ARQUITETURA_CICD.md](docs/ARQUITETURA_CICD.md). Em resumo:

- A cada pull request, o GitHub Actions executa lint, testes unitários das regras de negócio, testes de contrato da tabela Gold e validação da infraestrutura.
- A cada merge na branch principal, o Databricks Asset Bundles publica o job do pipeline (`01` a `05`), com ambientes de desenvolvimento e produção separados e execução semanal.
- O novo modelo só é promovido se a Precisão@3 for igual ou superior à da recomendação por popularidade; caso contrário, o modelo anterior é mantido.
- A infraestrutura (catálogo, schemas, volume e permissões) é definida em Terraform, e as credenciais ficam em GitHub Secrets.

## Limitações

- A maioria dos viajantes avaliou apenas 10 destinos, o que limita o aprendizado sobre cada pessoa.
- Destinos como Grécia, Itália e Japão agradam à maioria, o que reduz a margem de ganho sobre a recomendação por popularidade.
- A pergunta sobre regiões foi respondida por 21% dos viajantes.
- A estação do ano segue o calendário do hemisfério norte, já que a maioria dos respondentes é dos EUA. Por isso a aplicação exibe o período do ano (Dez–Fev, Mar–Mai, Jun–Ago, Set–Nov).
- O orçamento está em dólares, por pessoa e por noite.
- A base não contém atributos dos destinos, como clima e custo.

## Próximos passos

- Filtragem colaborativa a partir das avaliações.
- Inclusão de atributos dos destinos (clima por mês, custo médio, tipo de paisagem).
- Uso do feedback da aplicação para retreinar o modelo.
- Implementação do pipeline de CI/CD proposto.

## Equipe

- Renato de Oliveira de Souza
- Luciano Gomes da Silva
- José Carlos Marques

## Créditos

Dados: Stravl Travel Preference Data ([github.com/Stravl/stravl-data](https://github.com/Stravl/stravl-data)). Projeto desenvolvido para fins acadêmicos.
