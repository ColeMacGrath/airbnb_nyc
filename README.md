# Precio de Airbnb en Nueva York con MLflow

## Estructura

```
Actividad5/
├── datos/
│   ├── datos_ini/AB_NYC_2019.csv        datos originales sin modificar
│   └── datos_limp/airbnb_nyc_clean.csv  datos limpios
├── fuentes/
│   ├── entrena.ipynb   cuaderno: exploración, limpieza, entrenamiento y comparación
│   ├── datos_prep.py   obtención, verificación de hash y limpieza
│   └── train.py        entrenamiento, ajuste, pruebas estadísticas y registro en MLflow
├── evidencias/         gráficas, tablas y capturas del panel de MLflow
├── requirements.txt
├── README.md
└── CHANGELOG.md
```

## Dataset

| Campo | Valor |
|---|---|
| Nombre | New York City Airbnb Open Data 2019, archivo `AB_NYC_2019.csv` |
| Origen | Inside Airbnb, insideairbnb.com, publicado en Kaggle por Dgomonov |
| Copia usada | `https://raw.githubusercontent.com/ManarOmar/New-York-Airbnb-2019/master/AB_NYC_2019.csv` |
| Licencia | CC0 1.0 en la publicación de Kaggle; Inside Airbnb publica sus datos con CC BY 4.0 |
| Tamaño | 48,895 filas y 16 columnas |
| SHA-256 original | `e420db40ff10fcb40efc1b5b1648ee0b18a48f4e4537155cecc59fe95d18783a` |
| SHA-256 limpio v1.0.0 | `5d0e1e21a46edda72d2a4ebb735b094f8b70d0eeab9648918b4bb8ae18d8f88c` |
| Problema | Regresión: predecir `price` |

### Diccionario de datos

| Columna | Tipo | Descripción | Uso |
|---|---|---|---|
| id | entero | Identificador del anuncio | Se elimina por ser identificador |
| name | texto | Título del anuncio | Se elimina por ser texto libre |
| host_id | entero | Identificador del anfitrión | Se elimina por ser identificador |
| host_name | texto | Nombre del anfitrión | Se elimina por ser dato personal |
| neighbourhood_group | categórica | Distrito: Manhattan, Brooklyn, Queens, Bronx, Staten Island | Variable |
| neighbourhood | categórica | Vecindario, 221 valores | Variable |
| latitude, longitude | real | Coordenadas | Variables |
| room_type | categórica | Casa completa, cuarto privado o compartido | Variable |
| price | entero | Precio por noche en USD | Objetivo |
| minimum_nights | entero | Noches mínimas de estancia | Variable |
| number_of_reviews | entero | Total de reseñas | Variable |
| last_review | fecha | Fecha de la última reseña | Se transforma en `days_since_last_review` y `has_reviews` |
| reviews_per_month | real | Reseñas por mes | Variable |
| calculated_host_listings_count | entero | Anuncios del mismo anfitrión | Variable |
| availability_365 | entero | Días disponibles en el año | Variable |

### Supuestos

1. El precio publicado es el precio por noche en USD y es comparable entre anuncios.
2. Un anuncio sin reseñas tiene 0 reseñas por mes, no un dato perdido.
3. Precios menores a 10 USD o mayores a 1,000 USD son errores de captura o casos fuera del mercado objetivo. Son el 0.5% de los datos.
4. Más de 365 noches mínimas no tiene sentido comercial; se recorta a 365.
5. La fecha de referencia para la antigüedad de la última reseña es la fecha máxima del dataset, 2019-07-08.

## Reglas de limpieza

| Regla | Criterio | Filas afectadas |
|---|---|---|
| Espacios en texto | `strip` y espacios múltiples a uno | Todas las columnas de texto |
| Duplicados exactos | Fila idéntica | 0 |
| Duplicados por id | Mismo `id` | 0 |
| Duplicados lógicos | Mismo host, coordenadas, tipo y precio; se conserva el de más reseñas | 1 |
| Precio fuera de rango | Fuera de 10 a 1,000 USD | 250 eliminadas |
| Noches mínimas | Recorte a 365 | 14 recortadas |
| Nulos en `reviews_per_month` | 0 cuando no hay reseñas | 10,052 |
| Nulos en `last_review` | Días desde la última reseña con el valor máximo y bandera `has_reviews` | 10,052 |
| Nulos en `name` y `host_name` | "unknown" antes de eliminar la columna | 16 y 21 |

Los 113 posibles duplicados por anfitrión, nombre y precio se revisaron y se conservan: son unidades distintas de un mismo hotel u operador, con coordenadas y reseñas propias.

La codificación y el escalado no se guardan en el CSV limpio. Viven en el pipeline de entrenamiento, con `StandardScaler` para numéricas y `OneHotEncoder` con `min_frequency=50` para categóricas, para que se ajusten solo con datos de entrenamiento y no haya fuga de información.

## Modelos

| Modelo | Hardware | Hiperparámetros evaluados |
|---|---|---|
| Ridge | CPU, 10 núcleos en paralelo | `alpha`: 0.01, 0.1, 1, 10, 100 |
| XGBoost | CPU, 10 núcleos en paralelo | `n_estimators`: 300, 600, 1000; `max_depth`: 4, 6, 8, 10; `learning_rate`: 0.05, 0.1; `subsample`: 0.8, 1.0 |
| MLP en PyTorch | GPU Apple M1 Max con MPS | `hidden`: 64, 128, 256; `lr`: 0.001, 0.003 |

- Objetivo transformado con `log1p(price)` por la asimetría del precio; las métricas se reportan en USD.
- Hold-out 80/20 para la evaluación final y `KFold(10, shuffle=True, random_state=42)` sobre el 80% de entrenamiento. Los tres modelos usan los mismos folds, lo que permite pruebas pareadas.
- Métricas: MAE, RMSE y R².
- Comparación estadística: Friedman para los tres modelos, y prueba t pareada y Wilcoxon por pares. Se usan 10 folds porque con 5 el valor p mínimo posible de Wilcoxon es 0.0625 y nunca sería significativo.

## Reproducción

Requisitos: Python 3.14  y en macOS `brew install libomp` para XGBoost.

```bash
python3.14 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python fuentes/datos_prep.py      # descarga si falta, valida hash y genera el CSV limpio
python fuentes/train.py           # entrena, ajusta, compara y registra en MLflow
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

También se puede ejecutar todo desde `fuentes/entrena.ipynb` con Jupyter. El cuaderno usa las mismas funciones de los `.py`.

Notas:

- `train.py` importa `xgboost` antes que `torch`. En macOS, si torch se carga primero, sus runtimes de OpenMP chocan y XGBoost termina con un segmentation fault.
- Los modelos se guardan en MLflow con `cloudpickle` porque el formato por defecto, skops, no admite XGBoost ni la clase de la MLP. Solo se deben cargar modelos de este mismo experimento.
- `mlflow.db` y `mlruns/` no se versionan en Git; se regeneran al ejecutar el entrenamiento. La evidencia queda en `evidencias/`.

## MLflow

- Tracking local: `sqlite:///mlflow.db`, artefactos en `mlruns/`.
- Experimento: `Actividad5-Airbnb-NYC`.
- Un run padre por modelo con un run hijo por cada combinación del grid, más un run `Comparacion` con las pruebas estadísticas.
- Cada run guarda parámetros como el grid, los mejores valores, la semilla, los folds y la versión y hash del dataset. También guarda métricas de CV y de test, los resultados de CV, las gráficas y el modelo con su firma.
- El mejor modelo por RMSE en test se registra como `airbnb-nyc-price` con el alias `champion`.
