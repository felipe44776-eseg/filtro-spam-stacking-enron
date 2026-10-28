# Contrato de resultados: `resultados/metricas_finais.json`

**Para que serve.** README, notebook-história, os dois decks e a demo **só leem** números deste arquivo. Ele é gerado pelos scripts (`scripts/09_modelos_finais.py` consolida os demais). Se um número mudar, roda-se o script de novo e todos os artefatos se atualizam, sem nada digitado à mão.

## Regras
1. **Unidades:**
   - MCC, PR-AUC e F1 vão de 0 a 1.
   - **Taxas são frações de 0 a 1**, nunca porcentagens: `real_lixeira`, `spam_passou`.
   - O custo vem por mil e-mails: `custo_por_mil = 1000 × (9 × FP + 1 × FN) / n`.
   - Tempos em segundos.
2. **Sem arredondamento no arquivo.** Quem exibe arredonda: 3 casas para MCC; 1 casa para porcentagens e custo.
3. **Toda seção traz `fonte`:** o script e o arquivo de origem, para auditoria.
4. **Vocabulário:** chaves em português, sem acento (`real_lixeira`, `spam_passou`, `email_real`).
5. **Versão:** `versao_contrato` sobe a cada mudança de chave. Os leitores conferem a versão e falham com uma mensagem clara.

## Versão 2 (2026-10-28, revisão crítica, D11)

Só acréscimos; as chaves da versão 1 continuam. As chaves `finais.generalista` e `finais.generalista_sem_chave` guardam os nomes antigos: hoje exibem-se como "Stacking 8 (+TabPFN)" e "Stacking 7 padrão" (campo `rotulo`).

| Chave | O que é |
|---|---|
| `finais.tabpfn`, `finais.nao_filtrar` | o TabPFN sozinho e a referência "não filtrar", no mesmo formato dos outros filtros |
| `finais.*.rotulo` | o nome a exibir |
| `finais.*.{cv20, externo, spamassassin, teste_trancado}.{precisao_custo, revocacao_custo, f1_custo, tcr}` | precisão, revocação e F1 do spam no limiar de custo; TCR = custo de não filtrar ÷ custo do filtro |
| `finais.*.teste_trancado.matriz_confusao` | `spam_na_lixeira`, `real_na_caixa`, `real_na_lixeira`, `spam_na_caixa` |
| `comparacoes_custo_20_folds` | pares de filtros no custo dos 20 folds: diferença, IC95 corrigido, redução relativa, p de Holm e teste bayesiano (ROPE de 3 por mil) |
| `protocolo.teste_trancado.usos_descricao` | os 4 usos da base trancada |
| `genetico.ganho_sementes_novas_ic95`, `genetico.fracao_otimismo_ic95`, `genetico.comparacoes_custo`, `genetico.piso_par`, `genetico.piso_poder` | intervalo do ganho do GA, comparações de custo e a definição do piso (80% de poder, par GA − padrão) |
| `contra_estudo.permutacao.{n_permutacoes, modelo, mcc_permutado_media, mcc_permutado_max}` e `contra_estudo.quase_copias` | o teste de sinal com 999 permutações e o efeito das quase-cópias |
| `terceira_base.{nao_filtrar_custo, sensibilidade_custo, por_pasta, membros, conversor_spam, fontes_spam, semelhanca_spam}` | a referência, a sensibilidade a λ, os erros por pasta e por membro, a validação do conversor no spam e a fonte do spam de cada caixa |

Em `terceira_base.metodos`, o método "Stacking 7 padrão (Generalista atual)" passou a se chamar "Stacking 7 padrão", e entrou a linha "Não filtrar (referência)".

## Esquema original (versão 1, ilustrativo)

```json
{
  "versao_contrato": 1,
  "meta": {"gerado_em": "ISO-8601", "versao_pacote": "0.1.0", "python": "3.11.9", "seed_teste": 123,
           "custo_fp": 9, "custo_fn": 1},

  "base": {"n_emails": 5172, "n_email_real": 3672, "n_spam": 1500, "n_palavras": 3000,
           "fracao_zeros": 0.94, "duplicatas_excedentes": 541, "fracao_duplicatas_email_real": 0.93,
           "origem": "Enron1 (farmer-d + GP)", "contagem": "substring, sensível a maiúsculas", "fonte": "..."},

  "protocolo": {"teste_trancado": {"n": 1035, "email_real": 735, "spam": 300, "usos": 2},
                "folds_confirmacao": 20, "piso_mcc_detectavel": {"5": 0.0106, "20": 0.0086, "1000": 0.0079},
                "fonte": "..."},

  "metricas_didaticas": {"dummy_acuracia": 0.71, "exemplo_matriz": {"vp": 1484, "fp": 68, "fn": 16, "vn": 3604}},

  "nao_supervisionado": {"svd2d": [[x, y, rotulo]], "kmeans_ari": 0.0, "hierarquico_ari": 0.0,
                         "nmf_topicos": [{"topico": 1, "palavras": ["..."]}],
                         "anomalia_so_email_real": {"isolation_forest_mcc": 0.0, "one_class_svm_mcc": 0.0}, "fonte": "..."},

  "torneio": [{"modelo": "LinearSVC", "familia": "linear", "mcc": 0.963, "mcc_dp": 0.0, "pr_auc": 0.0,
               "real_lixeira": 0.0, "spam_passou": 0.0, "treino_s": 0.0}],
  "achados": {"tfidf_log": {"lr_mcc_antes": 0.922, "lr_mcc_depois": 0.961, "knn_mcc_antes": 0.75, "knn_mcc_depois": 0.94},
              "votacao_mcc": 0.959, "lr_mcc": 0.961},

  "stacking": {"variantes": [{"nome": "Stacking 7", "mcc": 0.975, "treino_s": 42.3}],
               "pesos_meta": {"kNN cosseno": 1.57},
               "erros_em_comum": {"modelos": ["..."], "matriz": [[0]]},
               "comparacao": {"contra": "LinearSVC", "p_holm": 0.17, "p_bayes_melhor": 0.774, "p_bayes_equiv": 0.224}},

  "genetico": {"combinacoes": 7077888, "avaliacoes": 857, "mcc_busca_padrao": 0.9732, "mcc_busca_ga": 0.9795,
               "mcc_busca_aleatoria": 0.9784, "ganho_busca": 0.0063, "ganho_sementes_novas": 0.0010,
               "curva_ga": [0.0], "curva_aleatoria": [0.0], "sensibilidade": {"kNN cosseno": 0.0055}},

  "corrida": {"candidatos": [{"nome": "#2", "mcc_corrida": 0.0, "mcc_verificacao": 0.0}], "vencedor_confirmou": false},

  "tabpfn": {"versao_nuvem": "TabPFN-3.5-Plus", "versao_local": "TabPFN-3.5-Fast", "creditos": 1490150,
             "interno": {"tabpfn_mcc": 0.9625, "stacking8_equiv_stacking7": 0.999},
             "externo": {"tabpfn_custo": 126.0, "stacking8_especialista_custo": 130.2},
             "local_x_nuvem": {"correlacao_notas": 0.0}},

  "finais": {
    "especialista": {"config": {"familia": "parametros"}, "limiar_custo": 0.0,
                     "cv20": {"mcc": 0.9744, "custo_por_mil": 36.6},
                     "teste_trancado": {"email_real_lixeira": 1, "spam_passou": 10, "custo": 19},
                     "externo": {"mcc": 0.894, "custo_por_mil": 207.6, "por_base": {"Enron2": {}}}},
    "generalista": {"...": "mesmo formato"},
    "linha_de_base": {"...": "LinearSVC, mesmo formato"}},

  "simulador_limiar": {"especialista": [{"limiar": 0.5, "real_lixeira": 0.0, "spam_passou": 0.0}],
                       "generalista": [{"limiar": 0.5, "real_lixeira": 0.0, "spam_passou": 0.0}]},

  "contra_estudo": {"permutacao_mcc": [0.0], "ablacao_palavras_enron": {"mcc_com": 0.0, "mcc_sem": 0.0},
                    "curva_aprendizado": [{"n": 250, "mcc": 0.0}], "sensibilidade_custo": [{"custo_fp": 1, "vencedor": "..."}]},

  "exemplos_demo": [{"rotulo": "spam", "texto": "...", "etapas": {"cobertura": 0.0, "notas_modelos": {}, "prob_final": 0.0}}]
}
```

Os valores acima são **ilustrativos**: os reais saem dos scripts. A ordem das seções segue o roteiro da história (`docs/ROTEIRO.md`, Fase 5).
