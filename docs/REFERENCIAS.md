# Referências

Agrupadas pelo uso no projeto. O texto integral dos artigos não é redistribuído (direito autoral): só a citação.

## Dados
- **Enron-Spam e o pré-processamento da AUEB.** V. Metsis, I. Androutsopoulos e G. Paliouras. *Spam Filtering with Naive Bayes – Which Naive Bayes?* Third Conference on Email and Anti-Spam (CEAS), 2006. Dados: http://www2.aueb.gr/users/ion/data/enron-spam/
- **Corpus Enron (origem dos e-mails reais).** B. Klimt e Y. Yang. *The Enron Corpus: A New Dataset for Email Classification Research*. ECML 2004, LNCS 3201, p. 217–226.
- **SpamAssassin public corpus** (versão 20030228). Apache SpamAssassin Project. https://spamassassin.apache.org/old/publiccorpus/
- **CSV do enunciado.** balaka18. *Email Spam Classification Dataset CSV*. Kaggle. https://www.kaggle.com/datasets/balaka18/email-spam-classification-dataset-csv

## Custo, λ = 9 e TCR
- I. Androutsopoulos, J. Koutsias, K. V. Chandrinos, G. Paliouras e C. D. Spyropoulos. *An Evaluation of Naive Bayesian Anti-Spam Filtering*. Workshop on Machine Learning in the New Information Age, 11th European Conference on Machine Learning (ECML), Barcelona, 2000, p. 9–17. Fonte do custo ponderado (λ), do cenário λ = 9 e do *total cost ratio* (TCR), que compara o filtro com não filtrar.

## Métricas
- B. W. Matthews. *Comparison of the predicted and observed secondary structure of T4 phage lysozyme*. Biochimica et Biophysica Acta 405(2), 1975, p. 442–451. (MCC)
- D. Chicco e G. Jurman. *The advantages of the Matthews correlation coefficient (MCC) over F1 score and accuracy in binary classification evaluation*. BMC Genomics 21, 6, 2020.
- L. Hubert e P. Arabie. *Comparing partitions*. Journal of Classification 2, 1985, p. 193–218. (ARI)

## Representação e não supervisionado
- G. Salton e C. Buckley. *Term-weighting approaches in automatic text retrieval*. Information Processing & Management 24(5), 1988, p. 513–523. (TF-IDF com log)
- S. Deerwester, S. T. Dumais, G. W. Furnas, T. K. Landauer e R. Harshman. *Indexing by latent semantic analysis*. Journal of the American Society for Information Science 41(6), 1990, p. 391–407. (SVD / LSA)
- D. D. Lee e H. S. Seung. *Learning the parts of objects by non-negative matrix factorization*. Nature 401, 1999, p. 788–791. (NMF)
- F. T. Liu, K. M. Ting e Z.-H. Zhou. *Isolation Forest*. IEEE ICDM 2008, p. 413–422.
- B. Schölkopf, J. C. Platt, J. Shawe-Taylor, A. J. Smola e R. C. Williamson. *Estimating the support of a high-dimensional distribution*. Neural Computation 13(7), 2001, p. 1443–1471. (One-Class SVM)

## Modelos
- J. D. M. Rennie, L. Shih, J. Teevan e D. R. Karger. *Tackling the Poor Assumptions of Naive Bayes Text Classifiers*. ICML 2003. (Complement Naive Bayes)
- S. Wang e C. D. Manning. *Baselines and Bigrams: Simple, Good Sentiment and Topic Classification*. ACL 2012. (NBSVM)
- P. Geurts, D. Ernst e L. Wehenkel. *Extremely randomized trees*. Machine Learning 63, 2006, p. 3–42.
- G. Ke et al. *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*. NeurIPS 2017.
- J. Platt. *Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods*. Advances in Large Margin Classifiers, MIT Press, 1999.
- N. Hollmann, S. Müller, L. Purucker et al. *Accurate predictions on small data with a tabular foundation model*. Nature 637, 2025, p. 319–326. (TabPFN; o projeto usa as versões 3.5-Plus, na nuvem, e 3.5-Fast, local, da Prior Labs: https://priorlabs.ai)

## Stacking, seleção e busca
- D. H. Wolpert. *Stacked generalization*. Neural Networks 5(2), 1992, p. 241–259.
- R. Caruana, A. Niculescu-Mizil, G. Crew e A. Ksikes. *Ensemble Selection from Libraries of Models*. ICML 2004.
- D. E. Goldberg. *Genetic Algorithms in Search, Optimization and Machine Learning*. Addison-Wesley, 1989.
- J. Bergstra e Y. Bengio. *Random Search for Hyper-Parameter Optimization*. Journal of Machine Learning Research 13, 2012, p. 281–305.
- G. C. Cawley e N. L. C. Talbot. *On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation*. Journal of Machine Learning Research 11, 2010, p. 2079–2107. (o otimismo de seleção, a "maldição do vencedor")

## Estatística
- C. Nadeau e Y. Bengio. *Inference for the Generalization Error*. Machine Learning 52(3), 2003, p. 239–281. (t corrigido)
- R. R. Bouckaert e E. Frank. *Evaluating the Replicability of Significance Tests for Comparing Learning Algorithms*. PAKDD 2004, LNCS 3056, p. 3–12. (validação cruzada repetida com o t corrigido)
- G. Corani e A. Benavoli. *A Bayesian approach for comparing cross-validated algorithms on multiple data sets*. Machine Learning 100, 2015, p. 285–304. (teste bayesiano correlacionado)
- A. Benavoli, G. Corani, J. Demšar e M. Zaffalon. *Time for a Change: a Tutorial for Comparing Multiple Classifiers Through Bayesian Analysis*. Journal of Machine Learning Research 18(77), 2017, p. 1–36. (ROPE; pacote `baycomp`)
- S. Holm. *A Simple Sequentially Rejective Multiple Test Procedure*. Scandinavian Journal of Statistics 6(2), 1979, p. 65–70.
- Q. McNemar. *Note on the sampling error of the difference between correlated proportions or percentages*. Psychometrika 12(2), 1947, p. 153–157.
- B. Efron e R. J. Tibshirani. *An Introduction to the Bootstrap*. Chapman & Hall, 1993.
