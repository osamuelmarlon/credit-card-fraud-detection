# Detecção de Fraudes em Cartões de Crédito

Pipeline completo de machine learning para detecção de transações
fraudulentas, com foco em boas práticas de modelagem e métricas
adequadas ao problema.

## Sobre o projeto

O dataset utilizado é o [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud),
disponível publicamente no Kaggle. Ele contém 284.807 transações reais
anonimizadas via PCA, com apenas 492 fraudes (0,17% do total).

O problema de negócio tem uma tensão central: detectar o máximo de
fraudes possível sem gerar alarmes falsos em excesso. Um modelo com
recall alto mas precision baixa é operacionalmente inviável. O
objetivo foi encontrar o equilíbrio certo entre os dois.

## Pipeline

| Etapa | Decisão |
|---|---|
| Separação dos dados | Train / Validation / Test com stratified split |
| Duplicatas | Investigadas e removidas (1.081 registros, incluindo fraudes) |
| Normalização | RobustScaler em `Amount` e `Time` |
| Desbalanceamento | `class_weight='balanced'` na Regressão Logística |
| Calibração | `CalibratedClassifierCV` com método isotônico |
| Threshold | Otimizado por F1-score na validação, fixo no teste |

## Principais desafios

**Desbalanceamento extremo:** 492 fraudes em 284.807 transações exigiu
tratamento cuidadoso em todas as etapas, desde o split estratificado
até a escolha das métricas de avaliação.

**Probabilidades mal calibradas:** sem calibração, a Regressão Logística
produzia probabilidades comprimidas e o threshold ótimo convergia para
0.99, inviável na prática. A calibração isotônica corrigiu isso e
trouxe o threshold para 0.45.

**Métricas enganosas:** acurácia e AUC-ROC são inadequadas para datasets
desbalanceados. O projeto foi avaliado por AUC-PR e F1-score, que
refletem a qualidade real das detecções da classe minoritária.

## Resultados no conjunto de teste

| Métrica | Valor |
|---|---|
| AUC-PR | 0.83 |
| AUC-ROC | 0.99 |
| Precision (fraude) | 91.94% |
| Recall (fraude) | 80.28% |
| F1-score (fraude) | 0.857 |
| Falsos positivos | 5 |
| Fraudes detectadas | 57 de 71 |

De 71 fraudes no conjunto de teste, o modelo detectou 57 gerando apenas
5 falsos alarmes. A cada 62 alertas disparados, 57 são fraudes reais.

## Tecnologias utilizadas

- Python 3
- pandas, numpy
- scikit-learn
- Regressão Logística
- XGBoost
- matplotlib, seaborn

## Como executar

1. Faça o download do dataset no
[Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
2. Coloque o arquivo `creditcard.csv` na mesma pasta do notebook
3. Execute as células em ordem

## Autor

[Samuel Costa](https://www.linkedin.com/in/samuel-marlon-lopes-costa/)
```
