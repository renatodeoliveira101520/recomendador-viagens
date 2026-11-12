Bússola — Recomendação Personalizada de Destinos de Viagem

Trabalho final da disciplina Hands-On Engenharia de Dados em Big Data e Analytics — MBA em Engenharia de Dados, Universidade Presbiteriana Mackenzie.

O sistema recebe as preferências de viagem de uma pessoa (faixa etária, orçamento, estação, experiências, paisagens, nível de atividade, segurança, popularidade e regiões) e devolve 3 destinos personalizados, com a probabilidade de interesse e o motivo de cada recomendação.

Base de dados

Usamos a base pública Stravl Travel Preference Data (github.com/Stravl/stravl-data), liberada para uso acadêmico e comercial.

Item	Quantidade
Viajantes no formulário	80.301
Viajantes com avaliações	61.130
Destinos avaliados	1.452
Avaliações (Sim / Não / Talvez)	813.802

Cada viajante respondeu a um formulário de preferências e depois avaliou destinos com Sim, Não ou Talvez.

Arquitetura (Databricks + Delta Lake)
Bronze  →  dados originais do Stravl (formulário, swipes, catálogo de destinos)
Silver  →  limpeza, tradução das respostas, nomes dos destinos, flags de qualidade
Gold    →  uma linha por avaliação (viajante × destino) + perfil traduzido
              ↓
        EDA  •  Modelo de recomendação  •  Top 3 por viajante (tabela Delta)

Tabela Gold (gold_stravl_v_2) — 813.802 linhas, 15 colunas: usuario_id, destino, pais, form_a_pt (faixa etária), form_b_pt (orçamento), form_c_pt (estação), form_f_pt (experiências), form_g_pt (paisagens), form_h_pt (nível de atividade), form_i_pt (segurança), form_j_pt (popularidade), form_r_pt (para onde quer ir), form_rr_pt (regiões), avaliacao, target (1 = Sim, 0 = Não, vazio = Talvez).

Estrutura do repositório
├── notebooks/
│   ├── stravl_eda.py                   # Análise exploratória
│   └── stravl_modelo_recomendacao.py   # Modelo, avaliação, formulário e Top 3
├── dados/
│   └── gold_stravl_v2.parquet          # Camada Gold pronta para carregar
└── README.md
Como executar
Criar a tabela Gold: no Databricks, em Catalog → + → Create table, envie gold_stravl_v2.parquet para workspace.default.
Importar os notebooks: em Workspace → ⋮ → Import.
Conferir a configuração: na primeira célula de cada notebook, TABELA_GOLD deve apontar para a tabela criada.
Executar: Run all (Serverless).
Testar o formulário: abra Show parameters, preencha as preferências e execute a célula "9. Resultado do formulário".
Análise exploratória — principais achados
22% dos viajantes recusaram todos os destinos que viram; apenas 4% aprovaram todos.
21% marcaram todas as faixas de orçamento e idade ao mesmo tempo (indício de que pularam a pergunta). Esses registros foram mantidos e sinalizados.
Destinos nos EUA concentram 40% das avaliações e têm a menor aceitação (21% de Sim, contra 47% no resto do mundo).
As preferências quase não mudam quanto as pessoas aprovam, mas mudam bastante quais destinos elas aprovam. Exemplos: quem busca praia aprova muito mais Belize e Bahamas; quem busca montanha, Füssen e Pokhara; quem busca deserto, destinos no Egito.
Cinco personas foram identificadas por K-Means (ex.: natureza e montanha, praia e mar, cultura urbana).
Modelo

Regressão Logística que estima a probabilidade de o viajante dizer "Sim" para cada destino.

Variável-alvo: Sim = 1, Não = 0 (as respostas "Talvez" ficam fora do treino).
Variáveis: cada resposta do formulário vira uma variável 0/1, e o modelo aprende um peso para cada combinação preferência × destino. Sem essas interações, o ranking seria igual para todos.
Validação: treino, validação e teste separados por viajante, ou seja, o modelo é avaliado em pessoas que ele nunca viu.
Regras aplicadas após o modelo
Contra os destinos campeões: o ranking usa probabilidade da pessoa − 0,5 × taxa de Sim geral do destino, para que destinos que agradam todo mundo não dominem as recomendações.
Diversidade: no máximo 1 destino por país no Top 3.
Região escolhida: se a pessoa seleciona regiões específicas, o Top 3 mostra apenas destinos dessas regiões.
Resultados (viajantes de teste)

O app mostra um Top 3. Escolhemos 3 posições por ser o melhor equilíbrio entre precisão e chance de acerto (comparação abaixo).

Métrica	Modelo (com regras)	Popularidade	Aleatório
Precisão@3 — dos 3 recomendados, quantos a pessoa aprovaria	52,7%	52,6%	36,2%
Acerto@3 — pessoas com pelo menos 1 "Sim" no Top 3	70,0%	69,3%	60,4%
Destinos diferentes recomendados (teste)	504	3	—
AUC	0,737	0,708	0,499
Acurácia	69,6%	—	63,4%*

* Acurácia de responder "Não" para todos.

Por que Top 3:

	Top 1	Top 3	Top 5
Precisão	57,2%	52,7%	48,6%
Acerto (≥ 1 "Sim")	57,2%	70,0%	73,1%

Do Top 3 para o Top 5, o acerto sobe só 3 pontos e a precisão cai 4.

As regras contra destinos campeões custam pouco: sem elas, a precisão@3 seria 53,9%, mas apenas 189 destinos diferentes apareceriam nas recomendações (contra 504 com as regras).

As métricas consideram apenas os cerca de 10 destinos que cada viajante avaliou, e não o catálogo inteiro — uma estimativa conservadora.

Limitações
A maioria dos viajantes avaliou apenas 10 destinos, o que limita o quanto o modelo aprende sobre cada pessoa.
O ganho sobre a recomendação por popularidade é pequeno: destinos como Grécia, Itália e Japão agradam a quase todos.
A pergunta de regiões foi respondida por apenas 21% dos viajantes.
A "estação" indica quando a pessoa viaja, no calendário do hemisfério norte (a maioria dos respondentes é dos EUA), e não o clima desejado.
O orçamento é em dólares, por pessoa e por noite, conforme o formulário original.
Próximos passos
Filtragem colaborativa usando os swipes ("quem recusou X também recusa Y").
Atributos dos destinos (clima por mês, custo médio, tipo de paisagem).
Uso do feedback dos usuários no app para re-treinar o modelo.

