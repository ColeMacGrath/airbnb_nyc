# CHANGELOG

Todas las versiones siguen versionado semántico. Los cambios del dataset y del código se registran por separado para que cada experimento de MLflow pueda ligarse a una versión exacta de datos mediante `data_version` y `data_sha256`.

## Dataset

### [data-1.0.0]

Primera versión limpia: `datos/datos_limp/airbnb_nyc_clean.csv`.

- SHA-256 original: `e420db40ff10fcb40efc1b5b1648ee0b18a48f4e4537155cecc59fe95d18783a`, con 48,895 filas y 16 columnas.
- SHA-256 limpio: `5d0e1e21a46edda72d2a4ebb735b094f8b70d0eeab9648918b4bb8ae18d8f88c`, con 48,644 filas y 13 columnas.
- Eliminado 1 duplicado lógico por mismo host, coordenadas, tipo de habitación y precio.
- Eliminados 250 anuncios con precio fuera de 10 a 1,000 USD.
- `minimum_nights` recortado a 365 en 14 registros.
- `reviews_per_month` nulo reemplazado por 0 cuando no hay reseñas.
- `last_review` convertido a `days_since_last_review` y `has_reviews`.
- Eliminadas columnas identificadoras y personales: `id`, `name`, `host_id`, `host_name`.
- Revisados y conservados 113 posibles duplicados por anfitrión y nombre.

## Código

### [1.0.0] 

- `datos_prep.py`: descarga con verificación SHA-256, limpieza con bitácora por paso y resumen de calidad antes y después.
- `train.py`: Ridge, XGBoost y MLP en PyTorch sobre MPS con `GridSearchCV` y `KFold(10)` compartido; métricas MAE, RMSE y R² en USD; Friedman, t pareada y Wilcoxon; registro en MLflow con runs padre e hijos, artefactos, firma del modelo y Model Registry con alias `champion`.
- `entrena.ipynb`: exploración, visualizaciones, entrenamiento y evidencia de MLflow usando los módulos anteriores.
- Grid de XGBoost ampliado a `max_depth` 10 y `n_estimators` 1000, y de la MLP a `hidden` 256, después de que la primera corrida dejara el óptimo en el borde del grid.
- Corregido un segmentation fault en macOS: `xgboost` se importa antes que `torch` y torch usa un solo hilo de CPU, porque ambos cargan su propio runtime de OpenMP.

### Entorno

- Python 3.14.8, porque 3.15 no tiene versiones de PyTorch ni de las dependencias de MLflow.
- Se reconstruyó el entorno virtual porque apuntaba a Python 3.13, que ya no estaba instalado.

## Reporte final

| Modelo | Mejores hiperparámetros | RMSE CV, media ± desv. | MAE test | RMSE test | R² test |
|---|---|---|---|---|---|
| Ridge | alpha = 0.01 | 94.72 ± 3.64 | 49.16 | 92.79 | 0.364 |
| XGBoost | learning_rate = 0.05, max_depth = 8, n_estimators = 600, subsample = 0.8 | 86.53 ± 3.62 | 44.36 | 85.09 | 0.465 |
| MLP en MPS | hidden = 256, lr = 0.003 | 91.63 ± 3.32 | 48.54 | 88.52 | 0.422 |

- Friedman: chi² = 20.0, p = 4.5e-05. XGBoost fue el mejor en los 10 folds y Ridge el peor en los 10.
- t pareada y Wilcoxon: las tres diferencias son significativas, con Wilcoxon p = 0.002 en todos los pares.
- Modelo registrado: `airbnb-nyc-price` v1 con XGBoost y alias `champion`.
- Las métricas están en USD por noche; el objetivo se entrenó como `log1p(price)`.
