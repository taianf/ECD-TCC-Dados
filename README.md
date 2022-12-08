# ECD-TCC-Dados

TCC da Pós ECD-UFBa

SRAG 2020:

- https://opendatasus.saude.gov.br/dataset/srag-2020

SRAG 2021-2022:

- https://opendatasus.saude.gov.br/dataset/srag-2021-e-2022

Ultima extração:

- 2022-10-12

Data dos arquivos:

- 2022-09-26

| Model                         | Training time | Models trained | Time per model     | F1                 |
|-------------------------------|---------------|----------------|--------------------|--------------------|
| Multilayer Perceptron         | 1759          | 18.0           | 97.72222222222223  | 0.5969435474385554 |
| Linear Support Vector Machine | 813           | 27.0           | 30.11111111111111  | 0.5964053797731378 |
| Gradient Boosted Tree         | 2082          | 27.0           | 77.11111111111111  | 0.5948102650730435 |
| Random Forest                 | 1002          | 45.0           | 22.266666666666666 | 0.5921345034100683 |
| Factorization Machines        | 1317          | 135.0          | 9.755555555555556  | 0.5828965711040142 |
| Logistic Regression           | 1140          | 135.0          | 8.444444444444445  | 0.5712969341163462 |
| Naive Bayes                   | 106           | 3.0            | 35.333333333333336 | 0.5434226785062449 |

## Regressão Logística

| Intercept           | Risk of death without any comorbidity (%) |
|---------------------|-------------------------------------------|
| -0.9615349252132588 | 27.65709825488868                         |

| Comorbidity                          | Logistic Regression Coefficient | Probability (%) compared to someone without comorbidities | Total Probability |
|--------------------------------------|---------------------------------|-----------------------------------------------------------|-------------------|
| Chronic Kidney Disease               | 0.4947658046512724              | 164.01                                                    | 45.36             |
| Postpartum                           | 0.4554332099218894              | 157.69                                                    | 43.61             |
| Immunodeficiency or Immunodepression | 0.44086903459070775             | 155.41                                                    | 42.98             |
| Chronic Liver Disease                | 0.43111209550372803             | 153.9                                                     | 42.56             |
| Diabetes Mellitus                    | 0.3855162684195613              | 147.04                                                    | 40.67             |
| Chronic Hematologic Disease          | 0.3315345906093188              | 139.31                                                    | 38.53             |
| Obesity                              | 0.30353342237041214             | 135.46                                                    | 37.46             |
| Asthma                               | 0.20758009401090677             | 123.07                                                    | 34.04             |
| Chronic Neurological Disease         | 0.19574036293390837             | 121.62                                                    | 33.64             |
| Other Chronic Pneumopathy            | 0.12989092233370036             | 113.87                                                    | 31.49             |
| Down's Syndrome                      | -0.14507414792834297            | 86.5                                                      | 23.92             |
| Chronic Cardiovascular Disease       | -0.3530147635129998             | 70.26                                                     | 19.43             |

