import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

class AdvancedTreeVisualizer:
    def __init__(self):
        self.model = None
        self.X = None
        self.y = None
        self.feature_names = None
        self.target_names = None
    
    def load_data(self):
        """Carga y prepara los datos"""
        iris = load_iris()
        self.X = iris.data
        self.y = iris.target
        self.feature_names = iris.feature_names
        self.target_names = iris.target_names
        
        # Entrenar modelo
        self.model = DecisionTreeClassifier(max_depth=3, random_state=42)
        self.model.fit(self.X, self.y)
    
    def create_3d_scatter(self):
        """Crea un gráfico 3D interactivo"""
        df = pd.DataFrame(self.X, columns=self.feature_names)
        df['species'] = [self.target_names[i] for i in self.y]
        
        fig = px.scatter_3d(
            df, 
            x=self.feature_names[0], 
            y=self.feature_names[1], 
            z=self.feature_names[2],
            color='species',
            title='🌸 Visualización 3D del Dataset Iris',
            labels={'species': 'Especie'},
            color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1']
        )
        
        fig.update_layout(
            scene=dict(
                xaxis_title=self.feature_names[0],
                yaxis_title=self.feature_names[1],
                zaxis_title=self.feature_names[2]
            ),
            font=dict(size=12)
        )
        
        fig.show()
    
    def create_decision_tree_sunburst(self):
        """Crea un gráfico sunburst del árbol de decisión"""
        tree = self.model.tree_
        
        # Extraer información del árbol
        def get_tree_data(node_id=0, parent="", depth=0):
            if tree.children_left[node_id] == tree.children_right[node_id]:
                # Nodo hoja
                class_counts = tree.value[node_id][0]
                predicted_class = np.argmax(class_counts)
                return [{
                    'ids': f"node_{node_id}",
                    'labels': f"Hoja: {self.target_names[predicted_class]}",
                    'parents': parent,
                    'values': tree.n_node_samples[node_id]
                }]
            else:
                # Nodo interno
                feature = tree.feature[node_id]
                threshold = tree.threshold[node_id]
                
                current_node = {
                    'ids': f"node_{node_id}",
                    'labels': f"{self.feature_names[feature][:10]}\n≤ {threshold:.2f}",
                    'parents': parent,
                    'values': tree.n_node_samples[node_id]
                }
                
                left_data = get_tree_data(
                    tree.children_left[node_id], 
                    f"node_{node_id}", 
                    depth + 1
                )
                right_data = get_tree_data(
                    tree.children_right[node_id], 
                    f"node_{node_id}", 
                    depth + 1
                )
                
                return [current_node] + left_data + right_data
        
        tree_data = get_tree_data()
        
        # Crear DataFrame
        df = pd.DataFrame(tree_data)
        
        fig = go.Figure(go.Sunburst(
            ids=df['ids'],
            labels=df['labels'],
            parents=df['parents'],
            values=df['values'],
            branchvalues="total",
            hovertemplate='<b>%{label}</b><br>Muestras: %{value}<extra></extra>',
            maxdepth=4
        ))
        
        fig.update_layout(
            title="🌞 Estructura del Árbol de Decisión (Sunburst)",
            font_size=12
        )
        
        fig.show()
    
    def create_animated_training(self):
        """Crea una animación del proceso de entrenamiento"""
        print("🎬 Creando animación del proceso de entrenamiento...")
        
        # Simular el proceso de entrenamiento con diferentes profundidades
        depths = range(1, 6)
        accuracies = []
        complexities = []
        
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=42
        )
        
        for depth in depths:
            model = DecisionTreeClassifier(max_depth=depth, random_state=42)
            model.fit(X_train, y_train)
            
            train_acc = model.score(X_train, y_train)
            test_acc = model.score(X_test, y_test)
            
            accuracies.append({
                'depth': depth,
                'train_accuracy': train_acc,
                'test_accuracy': test_acc,
                'n_nodes': model.tree_.node_count
            })
        
        df_acc = pd.DataFrame(accuracies)
        
        # Crear gráfico animado
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Precisión vs Profundidad', 'Número de Nodos', 
                          'Sobreajuste', 'Complejidad del Modelo'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Gráfico 1: Precisión
        fig.add_trace(
            go.Scatter(x=df_acc['depth'], y=df_acc['train_accuracy'],
                      mode='lines+markers', name='Entrenamiento',
                      line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df_acc['depth'], y=df_acc['test_accuracy'],
                      mode='lines+markers', name='Prueba',
                      line=dict(color='red')),
            row=1, col=1
        )
        
        # Gráfico 2: Número de nodos
        fig.add_trace(
            go.Bar(x=df_acc['depth'], y=df_acc['n_nodes'],
                  name='Nodos', marker_color='green'),
            row=1, col=2
        )
        
        # Gráfico 3: Diferencia (sobreajuste)
        overfitting = df_acc['train_accuracy'] - df_acc['test_accuracy']
        fig.add_trace(
            go.Scatter(x=df_acc['depth'], y=overfitting,
                      mode='lines+markers', name='Sobreajuste',
                      line=dict(color='orange')),
            row=2, col=1
        )
        
        # Gráfico 4: Complejidad
        fig.add_trace(
            go.Scatter(x=df_acc['n_nodes'], y=df_acc['test_accuracy'],
                      mode='markers', name='Complejidad vs Precisión',
                      marker=dict(size=10, color='purple')),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text="📊 Análisis del Entrenamiento del Árbol de Decisión",
            showlegend=True,
            height=800
        )
        
        fig.show()

def run_advanced_visualization():
    """Ejecuta las visualizaciones avanzadas"""
    print("🚀 VISUALIZACIONES AVANZADAS DE ÁRBOLES DE DECISIÓN")
    print("="*60)
    
    visualizer = AdvancedTreeVisualizer()
    visualizer.load_data()
    
    print("📊 Creando visualización 3D...")
    visualizer.create_3d_scatter()
    
    print("🌞 Creando gráfico sunburst...")
    visualizer.create_decision_tree_sunburst()
    
    print("🎬 Creando análisis de entrenamiento...")
    visualizer.create_animated_training()
    
    print("✅ ¡Visualizaciones avanzadas completadas!")

if __name__ == "__main__":
    run_advanced_visualization()
