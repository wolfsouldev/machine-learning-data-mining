import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, make_classification
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time

# Configuración de estilo
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class DecisionTreeVisualizer:
    def __init__(self):
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.target_names = None
        
    def load_iris_dataset(self):
        """Carga el dataset Iris para demostración"""
        print("🌸 Cargando dataset de flores Iris...")
        iris = load_iris()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            iris.data, iris.target, test_size=0.3, random_state=42
        )
        self.feature_names = iris.feature_names
        self.target_names = iris.target_names
        
        # Crear DataFrame para mejor visualización
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        df['species'] = [iris.target_names[i] for i in iris.target]
        
        print(f"✅ Dataset cargado: {len(df)} muestras")
        print(f"📊 Características: {list(self.feature_names)}")
        print(f"🎯 Clases: {list(self.target_names)}")
        
        return df
    
    def create_custom_dataset(self):
        """Crea un dataset personalizado más visual"""
        print("🎨 Creando dataset personalizado...")
        
        # Generar datos sintéticos
        X, y = make_classification(
            n_samples=200,
            n_features=2,
            n_redundant=0,
            n_informative=2,
            n_clusters_per_class=1,
            random_state=42
        )
        
        # Crear nombres más descriptivos
        feature_names = ['Característica X', 'Característica Y']
        target_names = ['Clase A', 'Clase B']
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        self.feature_names = feature_names
        self.target_names = target_names
        
        df = pd.DataFrame(X, columns=feature_names)
        df['clase'] = [target_names[i] for i in y]
        
        return df
    
    def train_decision_tree(self, max_depth=3):
        """Entrena el árbol de decisión"""
        print(f"🌳 Entrenando árbol de decisión (profundidad máxima: {max_depth})...")
        
        self.model = DecisionTreeClassifier(
            max_depth=max_depth,
            random_state=42,
            criterion='gini'
        )
        
        self.model.fit(self.X_train, self.y_train)
        
        # Evaluar el modelo
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        
        print(f"✅ Modelo entrenado con precisión: {accuracy:.2%}")
        return accuracy
    
    def visualize_tree_structure(self):
        """Visualiza la estructura del árbol"""
        print("📊 Creando visualización de la estructura del árbol...")
        
        plt.figure(figsize=(20, 12))
        plot_tree(
            self.model,
            feature_names=self.feature_names,
            class_names=self.target_names,
            filled=True,
            rounded=True,
            fontsize=10
        )
        plt.title("🌳 Estructura del Árbol de Decisión", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def visualize_decision_boundaries(self, df):
        """Visualiza las fronteras de decisión (solo para 2D)"""
        if len(self.feature_names) != 2:
            print("⚠️ Visualización de fronteras solo disponible para 2 características")
            return
            
        print("🎯 Creando visualización de fronteras de decisión...")
        
        # Crear malla de puntos
        h = 0.02
        x_min, x_max = self.X_train[:, 0].min() - 1, self.X_train[:, 0].max() + 1
        y_min, y_max = self.X_train[:, 1].min() - 1, self.X_train[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        
        # Predecir en la malla
        Z = self.model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Crear la visualización
        plt.figure(figsize=(12, 8))
        plt.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
        
        # Plotear puntos de entrenamiento
        scatter = plt.scatter(self.X_train[:, 0], self.X_train[:, 1], 
                            c=self.y_train, cmap='RdYlBu', edgecolors='black')
        
        plt.xlabel(self.feature_names[0], fontsize=12)
        plt.ylabel(self.feature_names[1], fontsize=12)
        plt.title('🎯 Fronteras de Decisión del Árbol', fontsize=14, fontweight='bold')
        plt.colorbar(scatter)
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def interactive_prediction_demo(self):
        """Demostración interactiva de predicciones"""
        print("🎮 Iniciando demostración interactiva...")
        
        # Tomar algunas muestras de prueba
        n_samples = min(5, len(self.X_test))
        
        for i in range(n_samples):
            sample = self.X_test[i:i+1]
            true_label = self.y_test[i]
            predicted_label = self.model.predict(sample)[0]
            probabilities = self.model.predict_proba(sample)[0]
            
            print(f"\n{'='*50}")
            print(f"🔍 MUESTRA {i+1}")
            print(f"{'='*50}")
            
            # Mostrar características
            for j, feature in enumerate(self.feature_names):
                print(f"📏 {feature}: {sample[0][j]:.2f}")
            
            print(f"\n🎯 Etiqueta real: {self.target_names[true_label]}")
            print(f"🤖 Predicción: {self.target_names[predicted_label]}")
            
            # Mostrar probabilidades
            print(f"\n📊 Probabilidades:")
            for j, prob in enumerate(probabilities):
                print(f"   {self.target_names[j]}: {prob:.2%}")
            
            # Mostrar el camino de decisión
            decision_path = self.model.decision_path(sample)
            leaf_id = self.model.apply(sample)
            
            print(f"\n🌳 Camino de decisión:")
            feature = self.model.tree_.feature
            threshold = self.model.tree_.threshold
            
            node_indicator = decision_path.toarray()[0]
            for node_id in range(len(node_indicator)):
                if node_indicator[node_id]:
                    if leaf_id[0] == node_id:
                        print(f"   🍃 Nodo hoja {node_id}: Predicción final")
                    else:
                        if sample[0][feature[node_id]] <= threshold[node_id]:
                            direction = "≤"
                        else:
                            direction = ">"
                        print(f"   🔀 Nodo {node_id}: {self.feature_names[feature[node_id]]} "
                              f"{direction} {threshold[node_id]:.2f}")
            
            input("\n⏸️  Presiona Enter para continuar...")
    
    def create_feature_importance_plot(self):
        """Crea gráfico de importancia de características"""
        print("📈 Creando gráfico de importancia de características...")
        
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(range(len(importances)), importances[indices], 
                      color='skyblue', edgecolor='navy', alpha=0.7)
        
        plt.title('📊 Importancia de las Características', fontsize=14, fontweight='bold')
        plt.xlabel('Características', fontsize=12)
        plt.ylabel('Importancia', fontsize=12)
        plt.xticks(range(len(importances)), 
                  [self.feature_names[i] for i in indices], rotation=45)
        
        # Añadir valores en las barras
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def print_tree_rules(self):
        """Imprime las reglas del árbol en texto"""
        print("📝 Reglas del árbol de decisión:")
        print("="*60)
        tree_rules = export_text(self.model, 
                                feature_names=self.feature_names,
                                class_names=self.target_names)
        print(tree_rules)

def main():
    """Función principal del programa"""
    print("🌟 VISUALIZADOR DE ÁRBOLES DE DECISIÓN 🌟")
    print("="*50)
    
    visualizer = DecisionTreeVisualizer()
    
    # Menú de opciones
    while True:
        print("\n🎯 OPCIONES DISPONIBLES:")
        print("1. 🌸 Usar dataset Iris (flores)")
        print("2. 🎨 Usar dataset personalizado (2D)")
        print("3. 🚪 Salir")
        
        choice = input("\n👉 Selecciona una opción (1-3): ").strip()
        
        if choice == '1':
            df = visualizer.load_iris_dataset()
            break
        elif choice == '2':
            df = visualizer.create_custom_dataset()
            break
        elif choice == '3':
            print("👋 ¡Hasta luego!")
            return
        else:
            print("❌ Opción inválida. Intenta de nuevo.")
    
    # Entrenar el modelo
    accuracy = visualizer.train_decision_tree(max_depth=4)
    
    # Ejecutar visualizaciones
    print("\n🎬 Iniciando visualizaciones...")
    
    # 1. Estructura del árbol
    visualizer.visualize_tree_structure()
    
    # 2. Fronteras de decisión (si es 2D)
    visualizer.visualize_decision_boundaries(df)
    
    # 3. Importancia de características
    visualizer.create_feature_importance_plot()
    
    # 4. Reglas del árbol
    visualizer.print_tree_rules()
    
    # 5. Demo interactiva
    demo_choice = input("\n🎮 ¿Quieres ver la demostración interactiva? (s/n): ").lower()
    if demo_choice == 's':
        visualizer.interactive_prediction_demo()
    
    print("\n🎉 ¡Visualización completada!")
    print("💡 El árbol de decisión toma decisiones siguiendo reglas simples")
    print("   basadas en las características de los datos.")

if __name__ == "__main__":
    main()
