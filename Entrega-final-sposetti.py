# --- Análisis de Dataset: Transacciones financieras con riesgo de lavado de dinero ---

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 

# --- Análisis Exploratorio de Datos ---

# Creo el DataFrame
ruta = r"C:\Users\kbm19\Desktop\Franco Cursos\Curso Data Science\Big_Black_Money_Dataset.csv"
data = pd.read_csv(ruta)

#  --- Limpieza de Datos ---

# Cambio el nombre de las columnas
data = data.rename(columns={
    "Index":"Indice",
    "Transaction ID": "Transaccion ID",
    "Country": "Pais",
    "Amount (USD)": "Monto",
    "Transaction Type": "Tipo de transaccion",
    "Date of Transaction": "Fecha",
    "Person Involved": "Persona involucrada",
    "Industry": "Industria",
    "Destination Country": "Pais destino",
    "Reported by Authority": "Declarado",
    "Source of Money": "Fuente del dinero",
    "Money Laundering Risk Score": "Riesgo de lavado",
    "Shell Companies Involved": "Empresa involucrada",
    "Financial Institution": "Institucion financiera",
    "Tax Haven Country;;;;": "Paraiso fiscal"
})
print("\n Tamaño del dataset antes de la limpieza de datos :")
print(data.shape)

# Selecciono variables relevantes 
data = data.drop(columns=["Indice","Persona involucrada","Empresa involucrada","Institucion financiera"])

# Convierto la columna fecha en Datetime
data["Fecha"] = pd.to_datetime(data["Fecha"], errors="coerce")

# Normalizo los montos 
data["Monto"] = (data["Monto"] / 100).astype(int)

# -- Exploración Inicial del Dataset --
print("Primeras filas del dataset:")
print(data.head())

print("\n Información del Dataset:")
print(data.info())

print("\n Tamaño del dataset:")
print(data.shape)

print("\n Valores nulos del dataset:")
print(data.isnull().sum())

print("\n Resumen Estadístico de las variables numéricas:")
print(data.drop(columns="Fecha").describe())

# --- Exploración del Dataset ---
print("\n Monto por país")
monto_pais = data.groupby("Pais")["Monto"].sum().reset_index()
monto_pais_ordenado = monto_pais.sort_values(by="Monto", ascending=False)
print(monto_pais_ordenado)

print("\n Monto por tipo de Transacción")
monto_transaccion = data.groupby("Tipo de transaccion")["Monto"].sum().reset_index()
monto_transaccion_oordenado = monto_transaccion.sort_values(by="Monto", ascending=False)
print(monto_transaccion_oordenado)

print("\n Monto por industria")
monto_industria = data.groupby("Industria")["Monto"].sum().reset_index()
monto_industria_ordenado = monto_industria.sort_values(by="Monto", ascending=False)
print(monto_industria_ordenado)

print("\n Monto por País destino")
monto_Destino = data.groupby("Pais destino")["Monto"].sum().reset_index()
monto_Destino_ordenado = monto_Destino.sort_values(by="Monto", ascending=False)
print(monto_Destino_ordenado)



print("\n Descripción del riesgo ")
print(data["Riesgo de lavado"].describe())
print(data.groupby("Riesgo de lavado")["Monto"].count())

print("\n Origen del dinero ")
print(data.groupby("Fuente del dinero")["Monto"].count())

print("\n Declarado o no")
print(data.groupby("Declarado")["Monto"].count())


# Extraer el mes y el año de la fecha
data['Mes'] = data['Fecha'].dt.to_period('M')  # Obtiene el mes y el año

# Calcular la moda del riesgo de lavado por mes
moda_por_mes = data.groupby('Mes')['Riesgo de lavado'].agg(lambda x: x.mode()[0]).reset_index()

# Crear el gráfico
plt.figure(figsize=(12, 6))
plt.plot(moda_por_mes['Mes'].dt.strftime('%Y-%m'), moda_por_mes['Riesgo de lavado'], marker='o', color='coral')
plt.title('Moda del Riesgo de Lavado por Mes', fontsize=16)
plt.xlabel('Mes', fontsize=14)
plt.ylabel('Moda del Riesgo de Lavado', fontsize=14)
plt.xticks(rotation=45)
plt.grid()
plt.tight_layout()
plt.show()

# Mostrar la tabla de modas por mes
print(moda_por_mes)


# --- Primeras Visualizaciones --- 
# --- Gráficos con Matplotlib ---

# 1) Gráficos univariados: Histograma de monto y riesgo de lavado
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
# Gráfico de monto
monto_count = data["Monto"]
monto_count.plot(kind="hist", color="blue", bins=30 , edgecolor="black", ax=ax1)
ax1.set_title("Distribución del Monto")
ax1.set_xlabel("Monto (en miles)")
ax1.set_ylabel("Frecuencia")
# Gráfico de riesgo 
riesgo_count = data["Riesgo de lavado"]
riesgo_count.plot(kind="hist", color="coral", edgecolor="black", ax=ax2)
ax2.set_title("Distribución del Riesgo")
ax2.set_xlabel("Riesgo")
ax2.set_ylabel("Frecuencia")
plt.show()

# 2) Gráfico bivariado: Relación entre el monto de la transacción y el riesgo de lavado por quartiles
# Agrupo el monto en 10 quartiles
data["Monto por Quartil"] = pd.qcut(data["Monto"], q=10, labels=["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8", "Q9", "Q10"])
# Calculo el promedio de riesgos
promedio_riesgo = data.groupby("Monto por Quartil")["Riesgo de lavado"].mean().reset_index()
# Creo el gráfico de barras
plt.figure(figsize=(10, 5))
plt.bar(promedio_riesgo["Monto por Quartil"], promedio_riesgo["Riesgo de lavado"], alpha=0.7, color="coral")
# Títulos y etiquetas
plt.title('Relación entre Monto por Quartiles y Riesgo de Lavado promedio', fontsize=14)
plt.xlabel('Grupo de Monto (USD)', fontsize=12)
plt.ylabel('Riesgo de Lavado Promedio', fontsize=12)
# Ajustar tamaño de los ticks en el eje x
plt.xticks(fontsize=10, rotation=45)  
plt.yticks(fontsize=10)  
# Limitar el rango del eje y 
plt.ylim(0, 10)  
# Mostrar gráfico
plt.grid(axis='y')  # Mostrar la cuadrícula solo en el eje y
plt.tight_layout()  # Ajustar el layout para evitar recortes
plt.show()

# --- Gráfico libre con Matplotlib --- 
# Gráfico libre con Matplotlib: Boxplot de Monto por Industria
#plt.figure(figsize=(14, 6))
data.boxplot(column='Monto', by='Industria', grid=False, showfliers=True, vert=False, patch_artist=True)

# Configuración del gráfico
plt.title('Distribución de Montos por Industria', fontsize=14)
plt.suptitle('')  # Eliminar el título automático que agrega pandas
plt.xlabel('Monto (en miles)', fontsize=12)
plt.ylabel('Industria', fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()

# --- Gráficos con Seaborn ---

# 1. Gráfico univariado: Violin plot de la distribución de Monto
plt.figure(figsize=(10, 6))
sns.violinplot(x=data['Monto'], color='skyblue')
plt.title('Distribución de Monto (Violin Plot)', fontsize=14)
plt.xlabel('Monto (en miles)', fontsize=12)
plt.ylabel('Densidad', fontsize=12)
plt.show()

# 2. Gráfico bivariado: Scatter plot del promedio de Riesgo de Lavado por Tipo de Transacción
# Calcular el promedio de riesgo de lavado por tipo de transacción
promedio_riesgo_tipo = data.groupby('Tipo de transaccion')['Riesgo de lavado'].mean().reset_index()
# Crear un scatter plot para mostrar el promedio del riesgo de lavado
plt.figure(figsize=(12, 6))
sns.scatterplot(x='Tipo de transaccion', y='Riesgo de lavado', data=promedio_riesgo_tipo, s=100, color='coral', alpha=0.7)
# Configuración del gráfico
plt.title('Promedio de Riesgo de Lavado por Tipo de Transacción', fontsize=14)
plt.xlabel('Tipo de Transacción', fontsize=12)
plt.ylabel('Riesgo de Lavado Promedio', fontsize=12)
plt.xticks(rotation=45)  # Rotar las etiquetas del eje x para mejor legibilidad
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# 3. Matriz de correlación entre variables numéricas
# Convertir variables booleanas a enteros para la matriz de correlación
data['Declarado'] = data['Declarado'].astype(int)
# Calcular la matriz de correlación entre variables numéricas
correlation_matrix = data[['Monto', 'Riesgo de lavado', 'Declarado']].corr()
# Visualización de la matriz de correlación usando heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Matriz de Correlación entre Variables Numéricas')
plt.show()

# ---  Boxplots para análisis categórico --- 
# Boxplot para 'Monto' según 'Fuente del dinero'
plt.figure(figsize=(12, 6))
sns.boxplot(x="Fuente del dinero", y="Monto", data=data)
plt.xticks(ticks=[0, 1], labels=["Ilegal", "Legal"])
plt.title('Distribución del Monto según Fuente del Dinero')
plt.xlabel('Fuente del Dinero')
plt.ylabel('Monto (en miles)')
plt.show()

# Boxplot del Riesgo de Lavado por Fuente del Dinero
plt.figure(figsize=(12, 6))
sns.boxplot(x='Fuente del dinero', y='Riesgo de lavado', data=data)
plt.xticks(rotation=45)
plt.title('Riesgo de Lavado por Fuente del Dinero')
plt.show()

# Boxplot del Riesgo de Lavado por País Destino
plt.figure(figsize=(12, 6))
sns.boxplot(x='Pais destino', y='Riesgo de lavado', data=data)
plt.xticks(rotation=45)
plt.title('Riesgo de Lavado por País Destino')
plt.show()

# Boxplot del Riesgo de Lavado por Tipo de Transacción
plt.figure(figsize=(12, 6))
sns.boxplot(x='Tipo de transaccion', y='Riesgo de lavado', data=data)
plt.xticks(rotation=45)
plt.title('Riesgo de Lavado por Tipo de Transacción')
plt.show()

# Boxplot del Riesgo de Lavado por Industria
plt.figure(figsize=(12, 6))
sns.boxplot(x='Industria', y='Riesgo de lavado', data=data)
plt.xticks(rotation=45)
plt.title('Riesgo de Lavado por Industria ')
plt.show()


import os
import warnings

# Configurar el número de núcleos si conoces la cantidad física (opcional)
os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # Reemplaza "4" por el número de núcleos físicos de tu máquina

# Ignorar advertencias
warnings.filterwarnings("ignore")

import os
import warnings
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, adjusted_rand_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.cluster import KMeans, DBSCAN

warnings.filterwarnings("ignore")

# Selección de variables
data_model = data[['Pais', 'Monto', 'Declarado', 'Fuente del dinero', 'Riesgo de lavado']]
X = data_model.drop(columns='Riesgo de lavado')
y = data_model['Riesgo de lavado']

# Aplicar LabelEncoder a las etiquetas (y)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  # Reindexar las clases para que comiencen desde 0

# Preprocesamiento: One-Hot Encoding y escalado
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), ['Pais', 'Fuente del dinero', 'Declarado']),
        ('num', StandardScaler(), ['Monto'])
    ]
)

# Configuración para PCA sin especificar svd_solver
pca = PCA(n_components=4)  # Ajustar n_components según el contexto de variabilidad deseada

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Modelos a evaluar
modelos = {
    "Regresión Logística": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(random_state=42)
}

# Parámetros para cada modelo en la búsqueda de hiperparámetros
parametros = {
    "Regresión Logística": {'classifier__C': [0.1, 1.0, 10]},
    "Random Forest": {'classifier__n_estimators': [50, 100, 200], 'classifier__max_depth': [None, 10, 20]},
    "XGBoost": {'classifier__learning_rate': [0.01, 0.1, 0.2], 'classifier__max_depth': [3, 6, 10]}
}

# Evaluación de modelos supervisados
for nombre, modelo in modelos.items():
    print(f"\n--- Resultados para {nombre} ---")
    
    # Pipeline con preprocesador, PCA y el modelo
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', modelo)
    ])
    
    # RandomizedSearchCV con búsqueda aleatoria de hiperparámetros
    grid = RandomizedSearchCV(pipeline, param_distributions=parametros[nombre], n_iter=5, cv=3, scoring='roc_auc', random_state=42)
    grid.fit(X_train, y_train)
    
    # Predicción y métricas
    y_pred = grid.predict(X_test)
    y_pred_proba = grid.predict_proba(X_test)[:, 1] if len(set(y)) == 2 else grid.predict_proba(X_test)  # Para binario o multiclase

    # Resultados y métricas
    print(classification_report(y_test, y_pred))

# Modelos no supervisados con DBSCAN y K-Means
print("\n--- Resultados para Modelos No Supervisados ---")

# Preprocesar y reducir dimensionalidad para DBSCAN y K-Means
X_preprocessed = preprocessor.fit_transform(X)
X_pca = pca.fit_transform(X_preprocessed)

# K-Means
kmeans = KMeans(n_clusters=10, random_state=42)
kmeans_clusters = kmeans.fit_predict(X_pca)
print("Índice de Rand Ajustado (K-Means):", adjusted_rand_score(y_encoded, kmeans_clusters))

# DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_clusters = dbscan.fit_predict(X_pca)
print("Índice de Rand Ajustado (DBSCAN):", adjusted_rand_score(y_encoded, dbscan_clusters))
