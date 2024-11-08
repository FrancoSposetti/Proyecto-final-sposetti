# Análisis de Transacciones Financieras con Riesgo de Lavado de Dinero

Este proyecto tiene como objetivo analizar un conjunto de datos de transacciones financieras relacionadas con el riesgo de lavado de dinero. El enfoque incluye tanto un análisis exploratorio de datos (EDA) para obtener insights iniciales, como la implementación de modelos de **Machine Learning** para realizar predicciones sobre el riesgo asociado a las transacciones.

## Contenido

- [Descripción del Dataset](#descripción-del-dataset)
- [Objetivos del Proyecto](#objetivos-del-proyecto)
- [Análisis Exploratorio de Datos (EDA)](#análisis-exploratorio-de-datos-eda)
- [Machine Learning](#machine-learning)
- [Resultados](#resultados)
- [Conclusiones](#conclusiones)
- [Cómo Ejecutar el Proyecto](#cómo-ejecutar-el-proyecto)

## Descripción del Dataset

El dataset utilizado en este proyecto se encuentra en el siguiente enlace: [Big_Black_Money_Dataset.csv](https://github.com/FrancoSposetti/pre-entrega1-sposetti/blob/main/Big_Black_Money_Dataset.csv). Este archivo contiene información sobre transacciones financieras, incluyendo variables como:

- ID de Transacción
- País
- Monto (en USD)
- Tipo de Transacción
- Fecha de la Transacción
- Riesgo de Lavado
- Fuente del Dinero
- Otros detalles relevantes

## Objetivos del Proyecto

1. Realizar una limpieza y preprocesamiento de los datos.
2. Realizar un análisis exploratorio de datos (EDA) para identificar patrones y tendencias.
3. Implementar y evaluar modelos de **Machine Learning** para predecir el riesgo de lavado de dinero.
4. Visualizar los resultados utilizando gráficos.
5. Determinar la relación entre las variables y el riesgo de lavado de dinero.

## Análisis Exploratorio de Datos (EDA)

En esta sección, se presentan los pasos llevados a cabo durante el análisis exploratorio de datos:

- **Carga del Dataset**: Se carga el dataset y se visualizan las primeras filas.
- **Limpieza de Datos**: Se lleva a cabo la limpieza de datos, incluyendo la transformación de nombres de columnas, manejo de valores nulos y conversión de tipos de datos.
- **Estadísticas Descriptivas**: Se analizan las variables numéricas mediante estadísticas descriptivas.
- **Agrupación de Datos**: Se realizan agrupaciones para calcular montos por país, tipo de transacción, y otras variables categóricas.
- **Visualización de Datos**: Se utilizan gráficos como histogramas, boxplots, scatter plots y heatmaps para observar distribuciones y relaciones.

## Machine Learning

En esta parte del proyecto, se aplicaron modelos de **Machine Learning** para predecir el riesgo de lavado de dinero:

- **Modelos Evaluados**: Se probaron los siguientes modelos:
  - **Regresión Logística**
  - **Random Forest**
  - **XGBoost**
  
- **Preprocesamiento**: Se realizó una combinación de **One-Hot Encoding** para variables categóricas y **Estandarización** para variables numéricas. Además, se utilizó **PCA (Análisis de Componentes Principales)** para reducir la dimensionalidad del conjunto de datos y mejorar la eficiencia del modelo.


- **Resultados obtenidos**:
  - **Regresión Logística**: El modelo mostró un rendimiento moderado, lo que sugiere que el modelo no tiene mucha capacidad de discriminación entre las clases.
  - **Random Forest**: Similar a la regresión logística, el rendimiento fue limitado con una precisión baja en la clasificación.
  - **XGBoost**: A pesar de ser un modelo potente, también obtuvo un rendimiento modesto en este conjunto de datos.

### Resultados de Modelos No Supervisados

Se probaron también modelos no supervisados como **K-Means** y **DBSCAN** para realizar agrupamientos, pero los resultados del **Índice de Rand Ajustado** fueron negativos, indicando que los clusters formados no tenían una relación significativa con las etiquetas de las clases.

## Resultados

Los resultados obtenidos en los modelos supervisados fueron los siguientes:

- **Regresión Logística**: 
    - Exactitud del 10% en la clasificación.
  
- **Random Forest**:
    - Exactitud del 9% en la clasificación.
  
- **XGBoost**:
    - Exactitud del 10% en la clasificación.
  
En los modelos **no supervisados** (K-Means y DBSCAN), el **Índice de Rand Ajustado** fue cercano a 0, lo que indica que no se encontraron patrones significativos en los datos para realizar agrupamientos.

## Conclusiones

1. **Modelos Supervisados**:
    - Los modelos aplicados (Regresión Logística, Random Forest y XGBoost) no lograron un desempeño destacado, con una baja precisión. Esto sugiere que el conjunto de datos podría tener un alto grado de complejidad, o que las características seleccionadas no son suficientes para predecir el riesgo de lavado de dinero de manera efectiva.
    - El bajo desempeño podría estar relacionado con el **desbalance de clases** o la **falta de características clave** para predecir con precisión los riesgos de lavado de dinero.

2. **Modelos No Supervisados**:
    - Tanto **K-Means** como **DBSCAN** mostraron resultados negativos en el **Índice de Rand Ajustado**, lo que indica que no se encontraron agrupamientos significativos en el conjunto de datos. Es posible que las clases no tengan una estructura natural de agrupamiento, o que los parámetros de los algoritmos no sean los más adecuados.

3. **Mejoras Futuras**:
    - Se podría realizar un **ajuste de hiperparámetros** más exhaustivo en los modelos supervisados, así como explorar técnicas de **balanceo de clases** como SMOTE.
    - También sería recomendable probar otros modelos de clasificación, como **SVM** o **Redes Neuronales**, y considerar **ajustes adicionales en el preprocesamiento** para mejorar la capacidad predictiva.
    - En el caso de los modelos no supervisados, se podrían probar otros algoritmos de clustering y ajustar sus parámetros.

## Cómo Ejecutar el Proyecto

Para ejecutar este proyecto, sigue estos pasos:

1. **Clona el repositorio**:
   ```bash
   git clone https://github.com/FrancoSposetti/Proyecto-final-sposetti.git

Ejecuta el notebook:

 Puedes abrir y ejecutar este proyecto en Google Colab usando el siguiente enlace:

 https://colab.research.google.com/drive/1mrvcGZDCNGnwGQLdFcsGhfW0g3LqLLNl?usp=sharing#scrollTo=TzP2PCoafEXwS