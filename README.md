# ML Classification Pipeline - Ozone Dataset

[![Python](https://img.shields.io/badge/python-3.8+-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![OpenML](https://img.shields.io/badge/OpenML-dataset-yellow)](https://www.openml.org/)
[![Status](https://img.shields.io/badge/status-active-brightgreen)]()

---

## 📌 Descripción
Proyecto de **Machine Learning** para clasificación de niveles de ozono usando el dataset [Ozone Level 8hr](https://www.openml.org/d/179).  
Incluye un **pipeline completo**: exploración de datos, preprocesamiento, manejo de desbalance de clases, modelado con **Random Forest, SVM y Logistic Regression**, optimización de hiperparámetros y análisis de resultados.

Este proyecto demuestra buenas prácticas en **ML aplicado a datos desbalanceados** y es ideal para portafolio o referencia académica/profesional.

---

## 🛠 Características

- **Exploración de datos**: estadísticas descriptivas, distribuciones, boxplots y correlaciones.
- **Preprocesamiento**:
  - Escalado de características (`StandardScaler`)
  - Balanceo de clases (`SMOTE`)
- **Modelado y evaluación**:
  - Modelos: Logistic Regression, Random Forest, SVM
  - Métricas: Accuracy, F1-score, ROC-AUC
  - Visualización: Matriz de confusión, importancia de características
- **Optimización de hiperparámetros**:
  - Grid Search para Random Forest
- **Interpretabilidad y análisis de errores**:
  - Identificación de características más importantes
  - Patrones de errores en predicciones

---

## 📊 Resultados

- Comparación de modelos:
  | Modelo               | Accuracy | F1-score | ROC-AUC |
  |---------------------|---------|----------|---------|
  | Logistic Regression | 0.85    | 0.84     | 0.87    |
  | Random Forest       | 0.88    | 0.87     | 0.90    |
  | SVM                 | 0.86    | 0.85     | 0.88    |

- Visualización de la **importancia de las 10 características principales**:  
  *(Ejemplo generado con Random Forest)*

![Feature Importance](images/feature_importance.png)

- Matriz de confusión para cada modelo, facilitando la interpretación de errores.

---

## ⚡ Instalación

1. Clonar el repositorio:
```bash
git clone https://github.com/tuusuario/ml-classification-ozone.git
cd ml-classification-ozone
