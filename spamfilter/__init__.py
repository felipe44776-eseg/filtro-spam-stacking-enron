"""spamfilter: o código reutilizável do projeto, a única fonte para scripts, notebooks e demo.

Módulos:
- caminhos: onde ficam dados, caches e resultados.
- dados: carga do CSV, grupos de cópias (linhas de contagem iguais), base de teste trancada, divisões sem vazamento.
- representacao: contagem por substring, TF-IDF com log, NBSVM.
- modelos: torneio de 22 modelos, os 13 de base, espaço de busca do Stacking 7 e os dois filtros finais.
- metricas: MCC, PR-AUC, e-mails reais na lixeira, spam que passou, custo e limiar de custo.
- estatistica: t corrigido, Holm, bayesiano correlacionado, McNemar, bootstrap por grupo.
- motor: notas out-of-fold em cache (o que tornou o algoritmo genético viável).
- genetico: algoritmo genético e busca aleatória de controle.
- externo: Enron1-6 da AUEB, vetorizados com o mesmo vocabulário.

Os números que estes módulos produzem são conferidos contra o registro de decisões pelos testes em tests/.
"""
__version__ = "0.1.0"
