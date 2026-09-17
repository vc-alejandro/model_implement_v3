# Análisis de Bias, Varianza y Ajuste del Modelo

Repositorio de la entrega: análisis y reporte sobre el desempeño de un
modelo, usando separación Train / Validation / Test, diagnóstico de
bias/varianza/ajuste, y mejora vía ajuste de hiperparámetros.

## Archivos

- **`decision_tree.py`**
  Implementación manual del árbol de decisión (ID3: entropía y ganancia
  de información), reutilizada de la entrega Parte I. Incluye una
  autoprueba con el dataset "Play Tennis" visto en clase, y corre solo
  con:
  ```
  python decision_tree.py
  ```

- **`WA_Fn-UseC_-Telco-Customer-Churn.csv`**
  Dataset Telco Customer Churn.

- **`bias_variance_analysis.ipynb`**
  Notebook con el análisis completo: separación train/validation/test,
  curva de complejidad (accuracy vs. max_depth) para diagnosticar bias
  y varianza, diagnóstico explícito con criterio de umbrales, búsqueda
  de hiperparámetros en dos etapas (exploratoria sobre una submuestra,
  fina sobre el train completo), y comparación final del modelo
  original contra el modelo ajustado en los tres conjuntos.

- **`reporte_bias_variance.docx`**
  Reporte con los resultados: separación de datos, diagnóstico de
  bias/varianza/ajuste (con criterio explícito), proceso de ajuste de
  hiperparámetros, y análisis/conclusión sobre la mejora obtenida.

- **`README.md`**
  Este archivo.

## Nota

Esta entrega reutiliza el árbol de decisión implementado manualmente
(no se modificó el algoritmo), enfocándose en el análisis de su
desempeño más que en la implementación.
