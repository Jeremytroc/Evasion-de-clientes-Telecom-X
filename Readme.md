# Telecom X - Modelo Predictivo de Evasión de Clientes (Churn)

## Descripción

Proyecto de ciencia de datos enfocado en la predicción de evasión de clientes (*churn*) para la empresa Telecom X. Se desarrolló un modelo de clasificación que identifica clientes con alta probabilidad de abandono, permitiendo a la empresa tomar acciones preventivas de retención.

El proyecto incluye un análisis exploratorio completo, pruebas estadísticas inferenciales, ingeniería de características y comparación de múltiples modelos de machine learning con optimización de hiperparámetros.

---

## Estructura del Proyecto

```
├── Telecom_X.ipynb       # Notebook principal con todo el análisis y modelado
├── data_limpia.csv        # Dataset preprocesado utilizado para el modelado
├── Readme.md              # Este archivo
```

---

## Metodología

### 1. Preparación de Datos
- Lectura del dataset limpio (`data_limpia.csv`)
- Eliminación de la columna `customerid`
- Separación de variables predictoras (X) y variable objetivo (y = `churn`)
- **One-Hot Encoding** para variables categóricas con `drop='if_binary'`
- **MinMaxScaler** para normalización de variables numéricas
- División en conjuntos de entrenamiento (80%) y prueba (20%)

### 2. Modelos Evaluados

| Modelo | Descripción |
|---|---|
| **Dummy Classifier** | Modelo baseline para establecer un umbral mínimo de rendimiento |
| **Random Forest** | Ensemble de árboles de decisión con `class_weight='balanced'` |
| **XGBoost** | Gradient Boosting acelerado por GPU (`device='cuda'`) con `scale_pos_weight` |

### 3. Selección de Features (Feature Importance)
- Se evaluó el rendimiento de cada modelo con diferentes cantidades de variables (16 a 21)
- Random Forest: **19 variables óptimas**
- XGBoost: **17 variables óptimas**
- Se seleccionaron **18 variables** como promedio razonable entre ambos modelos

### 4. Optimización de Hiperparámetros
- **GridSearchCV** con validación cruzada estratificada (5 folds)
- Métricas evaluadas: Accuracy, Precision, Recall y F1-Score
- Métrica de selección: **Recall** (prioridad en detectar clientes que abandonan)

#### Grilla de Random Forest
```
max_depth: [5, 10, 15]
min_samples_leaf: [1, 2, 13]
min_samples_split: [2, 4, 6]
n_estimators: [100, 150, 200]
```

#### Grilla de XGBoost
```
max_depth: [3, 5, 7]
n_estimators: [100, 150, 200]
learning_rate: [0.01, 0.1, 0.3]
subsample: [0.8, 1.0]
```

---

## Principales Hallazgos

1. **Cargos mensuales**: Los clientes que abandonan pagan significativamente más que los clientes activos.
2. **Antigüedad**: Los clientes nuevos tienen mayor probabilidad de abandono.
3. **Tipo de contrato**: Fuerte relación con la tasa de churn:
   - Mes a mes: **42.7%**
   - Anual: **11.3%**
   - Bianual: **2.8%**
4. **Tasa de churn general**: **26.6%**

---

## Resultados

### Modelo Ganador: Random Forest

El modelo de Random Forest con hiperparámetros optimizados obtuvo el mejor rendimiento en la métrica prioritaria (Recall), lo que permite detectar la mayor cantidad de clientes que realmente van a abandonar.

---

## Tecnologías Utilizadas

- **Python 3.10**
- **pandas** - Manipulación de datos
- **numpy** - Operaciones numéricas
- **matplotlib / seaborn** - Visualización
- **scikit-learn** - Modelos de ML, métricas, preprocesamiento y GridSearchCV
- **XGBoost** - Gradient Boosting con soporte GPU (CUDA)
- **yellowbrick** - Visualización de métricas de clasificación

---

## Cómo Ejecutar

1. Asegurarse de tener las dependencias instaladas:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost yellowbrick
   ```

2. Abrir `Telecom_X.ipynb` en VS Code o Jupyter Notebook.

3. Ejecutar todas las celdas en orden secuencial.

---

## Exportación del Modelo

El modelo ganador se puede exportar como archivo `.pkl` para su reutilización:

```python
import pickle

paquete = {
    'one_hot': one_hot,
    'features': features_seleccionadas,
    'modelo': modelo_optimo_forest.best_estimator_
}

with open('modelo_churn.pkl', 'wb') as f:
    pickle.dump(paquete, f)
```

---

## Autor

Proyecto desarrollado como parte del Postgrado en Ciencia de Datos - Estadísticas y Machine Learning.