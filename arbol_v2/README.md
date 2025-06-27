# 🌳 Visualizador de Árboles de Decisión

Un proyecto completo en Python para visualizar y entender cómo funcionan los algoritmos de árboles de decisión de manera interactiva y atractiva.

## 🌟 Características

- **Visualización de estructura del árbol**: Gráficos detallados de la estructura del árbol
- **Fronteras de decisión**: Visualización de cómo el árbol divide el espacio de características
- **Demo interactiva**: Seguimiento paso a paso de las decisiones del árbol
- **Gráficos 3D**: Visualizaciones tridimensionales interactivas
- **Análisis de importancia**: Gráficos de importancia de características
- **Múltiples datasets**: Iris y datasets personalizados

## 📋 Requisitos

- Python 3.8 o superior
- Las librerías listadas en `requirements.txt`

## 🚀 Instalación

1. **Clona o descarga el proyecto**
2. **Instala las dependencias**:
   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`

3. **Instala Graphviz** (necesario para visualizar árboles):
   
   **En Windows:**
   - Descarga desde: https://graphviz.org/download/
   - Añade Graphviz al PATH del sistema
   
   **En macOS:**
   \`\`\`bash
   brew install graphviz
   \`\`\`
   
   **En Ubuntu/Debian:**
   \`\`\`bash
   sudo apt-get install graphviz
   \`\`\`

## 🎮 Uso

### Visualización Básica
\`\`\`bash
python scripts/main_visualizer.py
\`\`\`

### Visualizaciones Avanzadas
\`\`\`bash
python scripts/advanced_visualizer.py
\`\`\`

## 📊 Funcionalidades

### 1. Visualización de Estructura
- Muestra la estructura completa del árbol
- Nodos coloreados por clase
- Información de división en cada nodo

### 2. Fronteras de Decisión
- Visualiza cómo el árbol divide el espacio
- Colores diferentes para cada región de decisión
- Puntos de datos superpuestos

### 3. Demo Interactiva
- Sigue el camino de decisión para muestras específicas
- Muestra probabilidades de cada clase
- Explica cada paso del proceso

### 4. Análisis de Importancia
- Gráfico de barras de importancia de características
- Ayuda a entender qué características son más relevantes

### 5. Visualizaciones 3D
- Gráficos tridimensionales interactivos
- Rotación y zoom disponibles
- Mejor comprensión de la distribución de datos

## 🎯 Datasets Disponibles

1. **Iris Dataset**: Clasificación de especies de flores
2. **Dataset Personalizado**: Datos sintéticos 2D para mejor visualización

## 🔧 Personalización

Puedes modificar los parámetros del árbol editando el archivo `main_visualizer.py`:

```python
# Cambiar profundidad máxima
visualizer.train_decision_tree(max_depth=5)

# Cambiar criterio de división
DecisionTreeClassifier(criterion='entropy')  # o 'gini'
