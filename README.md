**Materia:** Ciencia de Datos 2 · **Autor:** Santiago (Economía, UNAULA)

## Objetivo
Diseñar y ejecutar un experimento de **clasificación** que compare cuatro modelos clásicos con hiperparámetros por defecto (Regresión Logística, KNN, Árbol de Decisión y SVM-RBF), usando **validación cruzada** como herramienta principal de evaluación y desarrollando una mirada crítica sobre cómo seleccionar modelos.

## Dataset
**Breast Cancer Wisconsin (Diagnostic)** (UCI, también disponible en Kaggle; se carga con `sklearn.datasets.load_breast_cancer`).

| Característica | Detalle |
|---|---|
| Observaciones | 569 |
| Predictoras | 30, todas numéricas (radio, textura, perímetro, área, suavidad, concavidad, etc.: media, error estándar y peor valor) |
| Variable objetivo | Binaria: `1 = maligno` (37,3 %), `0 = benigno` (62,7 %) |
| Calidad | Sin valores faltantes, sin duplicados, sin categóricas |

## Metodología
1. Partición **estratificada 60/20/20** (entrenamiento / validación / prueba), `random_state=42`.
2. `Pipeline` de scikit-learn con `StandardScaler` + modelo (el escalado se reajusta en cada fold: sin fuga de información).
3. Para cada modelo: ROC AUC en entrenamiento y validación, y **CV estratificada de 5 folds** sobre entrenamiento (valor por fold, media y desviación estándar).
4. Análisis de robustez: repetición del procedimiento con 30 particiones aleatorias.
5. Modelo elegido → reentrenado con todo el entrenamiento → evaluado **una sola vez** en prueba (con IC bootstrap).

## Resultados

| Modelo | AUC Train | AUC Val | AUC CV (media ± std) |
|---|---|---|---|
| Regresión Logística | 0,998 | 0,996 | **0,994 ± 0,009** |
| SVM (RBF) | 0,997 | 0,996 | 0,994 ± 0,009 |
| KNN | 0,997 | 0,995 | 0,974 ± 0,040 |
| Árbol de Decisión | 1,000 | 0,897 | 0,936 ± 0,020 |

**Modelo final (Regresión Logística) en prueba:** AUC = **0,995** (IC 95 % bootstrap [0,985; 1,000]); 111/114 aciertos (sensibilidad 95,2 %, especificidad 98,6 %).

## Principales hallazgos y conclusiones
- **Regresión Logística ≈ SVM > KNN > Árbol de Decisión.** Las diferencias entre los dos primeros son mucho menores que la desviación estándar de la CV, por lo que se elige la logística por parsimonia e interpretabilidad.
- El **árbol sin poda sobreajusta** (AUC train = 1,0 vs. 0,897 en validación) y **KNN es inestable** entre folds (std = 0,040): la CV revela lo que una única validación oculta.
- El resultado en prueba es **consistente** con CV y validación: el procedimiento no sobreestimó el desempeño.
- **Limitaciones:** dataset "fácil" (AUC > 0,99), hiperparámetros por defecto (sin ajuste), prueba de solo 114 observaciones y sin validación externa; no apto para uso clínico real.

## Reproducibilidad
```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
jupyter notebook TP2_Aprendizaje_Supervisado.ipynb
```

## Contenido
- `TP2_Aprendizaje_Supervisado.ipynb` — notebook completo (ejecutado, con explicaciones, gráficos e interpretaciones).
- `README.md` — este archivo.
