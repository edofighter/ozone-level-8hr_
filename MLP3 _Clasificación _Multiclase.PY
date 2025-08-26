#librerías / Importación 
import openml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE

# Configuración de visualización
plt.style.use('ggplot')
sns.set_palette("Set2")

#Exploración de Datos
# Nota: El ID puede variar
datasets = openml.datasets.list_datasets(data_name='ozone-level-8hr')

#Primer resultado
dataset_id = list(datasets.keys())[0]

dataset = openml.datasets.get_dataset(dataset_id)
X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
df = pd.concat([X, y], axis =1)

# Información del dataset
print(f'Dimensiones del dataset: {df.shape}')
print('\nPrimeras 5 filas:')
print(df.head())
print('\nInformación del dataset:')
print(df.info())
print('\nEstadisticas descriptivas:')
print(df.describe())
print('\nDistribucion de la variable objetiva:')
print(y.value_counts())

# Análisis Exploratorio de Datos
# Configuración de visualizaciones
fig, axes = plt.subplots(2,2, figsize = (10,10))

# Distribución de la variable objetivo
sns.countplot(x=y, ax=axes[0,0])
axes[0,0].set_title('Distribución de Clases (Ozono)')
axes[0,0].set_xlabel('Clase')
axes[0,0].set_ylabel('Frecuencia')

# Matriz de correlación (solo columnas numéricas)
numeric_cols = df.select_dtypes(include=[np.number]).columns[:15]
cor_matrix = df[numeric_cols].corr()
sns.heatmap(cor_matrix, annot= True, fmt='.2f', cmap='coolwarm', ax = axes[0,1], 
            annot_kws={'size':6})
axes[0,1].set_title('Matriz de Correlación (Primeras 15 características)')

# Distribución de características importantes  para visualizar
sns.histplot(df['V1'], kde=True, ax=axes[1,0])
axes[1,0].set_title('Distribución de V1')

sns.histplot(df['V2'], kde=True, ax=axes[1,1])
axes[1,1].set_title('Distribución de V2')

plt.tight_layout()
#plt.show()

# Boxplots de relación entre características y la variable objetivo
fig, axes = plt.subplots(2,2, figsize = (15,12))
sns.boxplot(x=y, y='V1', data=df,ax=axes[0,0])
axes[0,0].set_title('V1 vs Clase de Ozono')

sns.boxplot(x=y, y='V2', data=df,ax= axes[0,1])
axes[0,1].set_title('V2 vs Clase de Ozono')

sns.boxplot(x=y, y='V3', data=df,ax= axes[1,0])
axes[1,0].set_title('V3 vs Clase de Ozono')

sns.boxplot(x=y, y='V4', data=df,ax= axes[1,1])
axes[1,1].set_title('V4 vs Clase de Ozono')

plt.tight_layout()
#plt.show()

# Análisis de correlación con la variable objetivo
corr_with_target = df.corr()[y.name].sort_values(ascending=False)
print('Características más correlacionadas con la variable objetivo:')
print(corr_with_target.head(10))

# Visualización de las  características más correlacionadas
plt.figure(figsize=(10,6))
sns.barplot(x=corr_with_target.head(10).values, y=corr_with_target.head(10).index)
plt.title('Top 10 Características Más Correlacionadas con la Clase')
plt.tight_layout()
#plt.show()

# Preprocesamiento de Datos  
# División de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Escalado de características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Manejo del desbalanceo de clases
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
print(f'Distribución antes de SMOTE: {np.bincount(y_train.astype(int))}')
print(f'Distribución después de SMOTE: {np.bincount(y_train_balanced.astype(int))}')

# Modelado y Evaluación
# Definición de modelos
models = {
 'Logistic Regression': LogisticRegression(
        random_state=42, 
        class_weight='balanced',
        max_iter=1000,  # Aumentar las iteraciones máximas
        solver='lbfgs'  # Explicitamente definir el solver
    ),
    'Random Forest': RandomForestClassifier(random_state=42, class_weight='balanced'),
    'SVM': SVC(random_state=42, class_weight='balanced', probability=True)
}

# Entrenamiento y evaluación de modelos
results = {}
for name, model in models.items():
    # Entrenamiento
    model.fit(X_train_balanced, y_train_balanced)
    
    # Predicción
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None 
    
    # Evaluación
    accuracy = accuracy_score(y_test, y_pred)
    f1 =  f1_score(y_test, y_pred, average= 'weighted')
    roc_auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None
    
    # Almacenar resultados
    results[name] = {
        'accuracy': accuracy,
        'f1_score' : f1,
        'roc_auc' : roc_auc,
        'model': model
    }
    
    # Reporte de clasificación
    print(f'\n{name} Classification Report:')
    print(classification_report(y_test, y_pred))
    
    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize= (6,4))
    sns.heatmap(cm, annot=True, fmt= 'd', cmap='Blues')
    plt.title (f'Matriz de Confusión - {name}')
    plt.ylabel('Verdadero')
    plt.xlabel('Predicción')
    plt.show()
    
# Comparación de modelos
results_df = pd.DataFrame(results).T
print('\nComparación de Modelos:')
print(results_df[['accuracy', 'f1_score', 'roc_auc']])


#Optimización de Hiperparámetros
# Grid reducido para pruebas rápidas
param_grid_rf_dict = {
    'n_estimators': [100],         # Solo un valor
    'max_depth' : [10, None],      # Menos opciones
    'min_samples_split' : [2],
    'min_samples_leaf' : [1]
}

rf = RandomForestClassifier(random_state=42, class_weight='balanced')
grid_search_rf = GridSearchCV(rf, param_grid_rf_dict, cv=3, scoring='f1_weighted', n_jobs=-1)
grid_search_rf.fit(X_train_balanced, y_train_balanced)
best_rf = grid_search_rf.best_estimator_
y_pred_rf = best_rf.predict(X_test_scaled)
feature_importances = best_rf.feature_importances_
feature_names = X.columns if hasattr(X, 'columns') else [f'V{i+1}' for i in range(X.shape[1])]

# Crear DataFrame para visualización
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance' : feature_importances
}).sort_values('importance', ascending=False) 
print("\nRandom Forest Optimizado - Classification Report:")
print(classification_report(y_test, y_pred_rf))

# Interpretación de Resultados
# Importancia de características (Random Forest)
feature_importances = best_rf.feature_importances_
feature_names = X.columns

# Crear DataFrame para visualización
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance' : feature_importances
}).sort_values('importance', ascending=False) 

# Visualizar las 10 características más importantes
plt.figure(figsize = (10,6))
sns.barplot(x= 'importance', y='feature', data = importance_df.head(10))
plt.title('Top 10 Características Más Importantes')
plt.tight_layout()
plt.show()

# Análisis de errores
errors = X_test.copy()
errors['true'] = y_test
errors['predicted'] = y_pred_rf
errors['error'] = errors['true'] != errors['predicted']

# Analizar patrones en los errores
print("\nEstadísticas de errores:")
print(f"Total de errores: {errors['error'].sum()}")
print(f"Tasa de error: {errors['error'].mean():.2%}")

# Características con mayor diferencia entre aciertos y errores
numeric_error_cols = errors.select_dtypes(include=[np.number]).columns
error_analysis = errors.groupby('error')[numeric_error_cols].mean().T
error_analysis['diferencia'] = abs(error_analysis[True] - error_analysis[False])
print("\nCaracterísticas con mayor diferencia entre aciertos y errores:")
print(error_analysis.sort_values('diferencia', ascending=False).head(10))
