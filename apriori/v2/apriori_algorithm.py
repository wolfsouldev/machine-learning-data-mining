import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
from itertools import combinations
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class AprioriVisualizer:
    def __init__(self, min_support=0.01, min_confidence=0.5):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.transactions = []
        self.frequent_itemsets = {}
        self.association_rules = []
        self.support_counts = {}
        
    def generate_large_dataset(self, n_transactions=10000, n_items=50):
        """Genera un dataset grande de transacciones de supermercado"""
        print(f"🔄 Generando dataset con {n_transactions} transacciones...")
        
        # Productos comunes en supermercado
        products = [
            'Pan', 'Leche', 'Huevos', 'Mantequilla', 'Queso', 'Yogurt', 'Pollo', 'Carne',
            'Pescado', 'Arroz', 'Pasta', 'Tomates', 'Cebollas', 'Papas', 'Zanahorias',
            'Manzanas', 'Plátanos', 'Naranjas', 'Fresas', 'Uvas', 'Cerveza', 'Vino',
            'Agua', 'Refrescos', 'Café', 'Té', 'Azúcar', 'Sal', 'Aceite', 'Vinagre',
            'Detergente', 'Jabón', 'Champú', 'Pasta_dental', 'Papel_higiénico',
            'Cereales', 'Galletas', 'Chocolate', 'Helado', 'Pizza', 'Sopa', 'Atún',
            'Jamón', 'Salchichas', 'Verduras_congeladas', 'Jugo', 'Yogurt_griego',
            'Queso_crema', 'Crema', 'Mantequilla_maní'
        ]
        
        # Patrones de compra realistas (algunos productos van juntos)
        common_patterns = [
            ['Pan', 'Mantequilla', 'Leche'],
            ['Huevos', 'Leche', 'Pan'],
            ['Pollo', 'Arroz', 'Verduras_congeladas'],
            ['Pasta', 'Tomates', 'Queso'],
            ['Cerveza', 'Papas', 'Salchichas'],
            ['Café', 'Azúcar', 'Leche'],
            ['Manzanas', 'Plátanos', 'Naranjas'],
            ['Detergente', 'Jabón', 'Papel_higiénico'],
            ['Cereales', 'Leche', 'Plátanos'],
            ['Pizza', 'Refrescos', 'Helado']
        ]
        
        transactions = []
        
        for i in range(n_transactions):
            transaction = []
            
            # 30% probabilidad de usar un patrón común
            if np.random.random() < 0.3:
                pattern = np.random.choice(len(common_patterns))
                transaction.extend(common_patterns[pattern])
            
            # Agregar productos aleatorios adicionales
            n_additional = np.random.poisson(3) + 1
            additional_products = np.random.choice(
                products, 
                size=min(n_additional, len(products)), 
                replace=False
            )
            transaction.extend(additional_products)
            
            # Remover duplicados y convertir a lista
            transaction = list(set(transaction))
            transactions.append(transaction)
        
        self.transactions = transactions
        print(f"✅ Dataset generado: {len(transactions)} transacciones")
        return transactions
    
    def calculate_support(self, itemset, transactions):
        """Calcula el soporte de un itemset"""
        count = 0
        for transaction in transactions:
            if set(itemset).issubset(set(transaction)):
                count += 1
        return count / len(transactions)
    
    def get_frequent_1_itemsets(self):
        """Encuentra itemsets frecuentes de tamaño 1"""
        print("🔍 Buscando itemsets frecuentes de tamaño 1...")
        
        # Contar frecuencia de cada item
        item_counts = Counter()
        for transaction in self.transactions:
            for item in transaction:
                item_counts[item] += 1
        
        # Filtrar por soporte mínimo
        frequent_1_itemsets = []
        total_transactions = len(self.transactions)
        
        for item, count in item_counts.items():
            support = count / total_transactions
            if support >= self.min_support:
                frequent_1_itemsets.append(([item], support))
                self.support_counts[frozenset([item])] = support
        
        print(f"✅ Encontrados {len(frequent_1_itemsets)} itemsets frecuentes de tamaño 1")
        return frequent_1_itemsets
    
    def generate_candidates(self, frequent_itemsets, k):
        """Genera candidatos de tamaño k a partir de itemsets frecuentes de tamaño k-1"""
        candidates = []
        n = len(frequent_itemsets)
        
        for i in range(n):
            for j in range(i + 1, n):
                # Unir dos itemsets si difieren en exactamente un elemento
                itemset1 = set(frequent_itemsets[i][0])
                itemset2 = set(frequent_itemsets[j][0])
                
                union = itemset1.union(itemset2)
                if len(union) == k:
                    candidates.append(sorted(list(union)))
        
        return candidates
    
    def apriori_algorithm(self):
        """Implementa el algoritmo Apriori completo"""
        print("🚀 Iniciando algoritmo Apriori...")
        
        # Paso 1: Encontrar itemsets frecuentes de tamaño 1
        frequent_itemsets = self.get_frequent_1_itemsets()
        self.frequent_itemsets[1] = frequent_itemsets
        
        k = 2
        while frequent_itemsets:
            print(f"🔍 Buscando itemsets frecuentes de tamaño {k}...")
            
            # Generar candidatos
            candidates = self.generate_candidates(frequent_itemsets, k)
            
            if not candidates:
                break
            
            # Calcular soporte para cada candidato
            frequent_k_itemsets = []
            for candidate in candidates:
                support = self.calculate_support(candidate, self.transactions)
                if support >= self.min_support:
                    frequent_k_itemsets.append((candidate, support))
                    self.support_counts[frozenset(candidate)] = support
            
            if not frequent_k_itemsets:
                break
            
            self.frequent_itemsets[k] = frequent_k_itemsets
            frequent_itemsets = frequent_k_itemsets
            print(f"✅ Encontrados {len(frequent_k_itemsets)} itemsets frecuentes de tamaño {k}")
            k += 1
        
        print(f"🎉 Algoritmo Apriori completado. Encontrados itemsets hasta tamaño {k-1}")
    
    def generate_association_rules(self):
        """Genera reglas de asociación a partir de itemsets frecuentes"""
        print("📋 Generando reglas de asociación...")
        
        rules = []
        
        for k in range(2, len(self.frequent_itemsets) + 1):
            if k not in self.frequent_itemsets:
                continue
                
            for itemset, support in self.frequent_itemsets[k]:
                itemset_set = set(itemset)
                
                # Generar todas las posibles reglas A -> B
                for i in range(1, len(itemset)):
                    for antecedent in combinations(itemset, i):
                        antecedent_set = set(antecedent)
                        consequent_set = itemset_set - antecedent_set
                        
                        if not consequent_set:
                            continue
                        
                        # Calcular confianza
                        antecedent_support = self.support_counts.get(frozenset(antecedent), 0)
                        if antecedent_support > 0:
                            confidence = support / antecedent_support
                            
                            if confidence >= self.min_confidence:
                                # Calcular lift
                                consequent_support = self.support_counts.get(frozenset(consequent_set), 0)
                                lift = confidence / consequent_support if consequent_support > 0 else 0
                                
                                rules.append({
                                    'antecedent': list(antecedent),
                                    'consequent': list(consequent_set),
                                    'support': support,
                                    'confidence': confidence,
                                    'lift': lift
                                })
        
        self.association_rules = sorted(rules, key=lambda x: x['confidence'], reverse=True)
        print(f"✅ Generadas {len(self.association_rules)} reglas de asociación")
        return self.association_rules
    
    def visualize_dataset_overview(self):
        """Visualiza información general del dataset"""
        print("📊 Creando visualización del dataset...")
        
        # Estadísticas básicas
        transaction_lengths = [len(t) for t in self.transactions]
        all_items = [item for transaction in self.transactions for item in transaction]
        item_frequencies = Counter(all_items)
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Distribución de Tamaño de Transacciones', 
                          'Top 20 Productos Más Frecuentes',
                          'Estadísticas del Dataset', 
                          'Distribución de Frecuencias'),
            specs=[[{"type": "histogram"}, {"type": "bar"}],
                   [{"type": "table"}, {"type": "histogram"}]]
        )
        
        # Histograma de tamaños de transacciones
        fig.add_trace(
            go.Histogram(x=transaction_lengths, name="Tamaño de Transacciones"),
            row=1, col=1
        )
        
        # Top productos más frecuentes
        top_items = dict(item_frequencies.most_common(20))
        fig.add_trace(
            go.Bar(x=list(top_items.keys()), y=list(top_items.values()), 
                   name="Frecuencia de Productos"),
            row=1, col=2
        )
        
        # Tabla de estadísticas
        stats_data = [
            ["Total de Transacciones", len(self.transactions)],
            ["Productos Únicos", len(item_frequencies)],
            ["Tamaño Promedio de Transacción", f"{np.mean(transaction_lengths):.2f}"],
            ["Tamaño Máximo de Transacción", max(transaction_lengths)],
            ["Tamaño Mínimo de Transacción", min(transaction_lengths)],
            ["Soporte Mínimo", f"{self.min_support:.3f}"],
            ["Confianza Mínima", f"{self.min_confidence:.3f}"]
        ]
        
        fig.add_trace(
            go.Table(
                header=dict(values=["Métrica", "Valor"]),
                cells=dict(values=[[row[0] for row in stats_data], 
                                 [row[1] for row in stats_data]])
            ),
            row=2, col=1
        )
        
        # Distribución de frecuencias
        frequencies = list(item_frequencies.values())
        fig.add_trace(
            go.Histogram(x=frequencies, name="Distribución de Frecuencias"),
            row=2, col=2
        )
        
        fig.update_layout(height=800, title_text="Análisis del Dataset de Transacciones")
        fig.show()
    
    def visualize_frequent_itemsets(self):
        """Visualiza los itemsets frecuentes por nivel"""
        print("📊 Visualizando itemsets frecuentes...")
        
        # Preparar datos para visualización
        levels = []
        counts = []
        itemsets_data = []
        
        for k, itemsets in self.frequent_itemsets.items():
            levels.append(f"Tamaño {k}")
            counts.append(len(itemsets))
            
            for itemset, support in itemsets[:10]:  # Top 10 por nivel
                itemsets_data.append({
                    'Nivel': k,
                    'Itemset': ' + '.join(itemset),
                    'Soporte': support,
                    'Tamaño': len(itemset)
                })
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Itemsets Frecuentes por Nivel', 
                          'Top Itemsets por Soporte',
                          'Distribución de Soporte', 
                          'Evolución del Algoritmo'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "histogram"}, {"type": "scatter"}]]
        )
        
        # Gráfico de barras por nivel
        fig.add_trace(
            go.Bar(x=levels, y=counts, name="Cantidad de Itemsets"),
            row=1, col=1
        )
        
        # Top itemsets por soporte
        df_itemsets = pd.DataFrame(itemsets_data)
        if not df_itemsets.empty:
            top_itemsets = df_itemsets.nlargest(15, 'Soporte')
            fig.add_trace(
                go.Bar(x=top_itemsets['Itemset'], y=top_itemsets['Soporte'],
                       name="Soporte", text=top_itemsets['Soporte'].round(3)),
                row=1, col=2
            )
            
            # Distribución de soporte
            fig.add_trace(
                go.Histogram(x=df_itemsets['Soporte'], name="Distribución de Soporte"),
                row=2, col=1
            )
            
            # Evolución del algoritmo
            fig.add_trace(
                go.Scatter(x=df_itemsets['Nivel'], y=df_itemsets['Soporte'],
                          mode='markers', name="Itemsets", 
                          text=df_itemsets['Itemset'],
                          marker=dict(size=8, opacity=0.6)),
                row=2, col=2
            )
        
        fig.update_layout(height=800, title_text="Análisis de Itemsets Frecuentes")
        fig.update_xaxes(tickangle=45, row=1, col=2)
        fig.show()
    
    def visualize_association_rules(self):
        """Visualiza las reglas de asociación"""
        print("📊 Visualizando reglas de asociación...")
        
        if not self.association_rules:
            print("❌ No hay reglas de asociación para visualizar")
            return
        
        # Preparar datos
        rules_df = pd.DataFrame(self.association_rules)
        rules_df['antecedent_str'] = rules_df['antecedent'].apply(lambda x: ' + '.join(x))
        rules_df['consequent_str'] = rules_df['consequent'].apply(lambda x: ' + '.join(x))
        rules_df['rule'] = rules_df['antecedent_str'] + ' → ' + rules_df['consequent_str']
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Top 15 Reglas por Confianza', 
                          'Soporte vs Confianza',
                          'Distribución de Lift', 
                          'Top 10 Reglas por Lift'),
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "histogram"}, {"type": "bar"}]]
        )
        
        # Top reglas por confianza
        top_rules = rules_df.head(15)
        fig.add_trace(
            go.Bar(x=top_rules['rule'], y=top_rules['confidence'],
                   name="Confianza", text=top_rules['confidence'].round(3)),
            row=1, col=1
        )
        
        # Scatter plot: Soporte vs Confianza
        fig.add_trace(
            go.Scatter(x=rules_df['support'], y=rules_df['confidence'],
                      mode='markers', name="Reglas",
                      text=rules_df['rule'],
                      marker=dict(size=rules_df['lift']*5, opacity=0.6,
                                colorscale='Viridis', color=rules_df['lift'],
                                colorbar=dict(title="Lift"))),
            row=1, col=2
        )
        
        # Distribución de Lift
        fig.add_trace(
            go.Histogram(x=rules_df['lift'], name="Distribución de Lift"),
            row=2, col=1
        )
        
        # Top reglas por Lift
        top_lift = rules_df.nlargest(10, 'lift')
        fig.add_trace(
            go.Bar(x=top_lift['rule'], y=top_lift['lift'],
                   name="Lift", text=top_lift['lift'].round(2)),
            row=2, col=2
        )
        
        fig.update_layout(height=800, title_text="Análisis de Reglas de Asociación")
        fig.update_xaxes(tickangle=45, row=1, col=1)
        fig.update_xaxes(tickangle=45, row=2, col=2)
        fig.show()
        
        # Mostrar tabla de mejores reglas
        print("\n🏆 TOP 10 REGLAS DE ASOCIACIÓN:")
        print("="*80)
        for i, rule in enumerate(rules_df.head(10).itertuples(), 1):
            print(f"{i:2d}. {rule.antecedent_str} → {rule.consequent_str}")
            print(f"    Soporte: {rule.support:.3f} | Confianza: {rule.confidence:.3f} | Lift: {rule.lift:.2f}")
            print()
    
    def create_network_visualization(self):
        """Crea una visualización de red de las reglas de asociación"""
        print("🕸️ Creando visualización de red...")
        
        if not self.association_rules:
            return
        
        import networkx as nx
        
        # Crear grafo
        G = nx.DiGraph()
        
        # Agregar nodos y aristas
        for rule in self.association_rules[:20]:  # Top 20 reglas
            antecedent = ' + '.join(rule['antecedent'])
            consequent = ' + '.join(rule['consequent'])
            
            G.add_node(antecedent, type='antecedent')
            G.add_node(consequent, type='consequent')
            G.add_edge(antecedent, consequent, 
                      weight=rule['confidence'], 
                      support=rule['support'],
                      lift=rule['lift'])
        
        # Posiciones de los nodos
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # Crear visualización con plotly
        edge_x = []
        edge_y = []
        edge_info = []
        
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            edge_data = G.edges[edge]
            edge_info.append(f"Confianza: {edge_data['weight']:.3f}<br>"
                           f"Soporte: {edge_data['support']:.3f}<br>"
                           f"Lift: {edge_data['lift']:.2f}")
        
        edge_trace = go.Scatter(x=edge_x, y=edge_y,
                               line=dict(width=2, color='#888'),
                               hoverinfo='none',
                               mode='lines')
        
        node_x = []
        node_y = []
        node_text = []
        node_info = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)
            
            # Información del nodo
            adjacencies = list(G.neighbors(node))
            node_info.append(f'{node}<br>Conexiones: {len(adjacencies)}')
        
        node_trace = go.Scatter(x=node_x, y=node_y,
                               mode='markers+text',
                               hoverinfo='text',
                               text=node_text,
                               textposition="middle center",
                               hovertext=node_info,
                               marker=dict(size=20,
                                         color='lightblue',
                                         line=dict(width=2, color='black')))
        
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title='Red de Reglas de Asociación',
                           titlefont_size=16,
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           annotations=[ dict(
                               text="Visualización de red mostrando relaciones entre productos",
                               showarrow=False,
                               xref="paper", yref="paper",
                               x=0.005, y=-0.002,
                               xanchor="left", yanchor="bottom",
                               font=dict(color="gray", size=12)
                           )],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
        
        fig.show()
    
    def run_complete_analysis(self):
        """Ejecuta el análisis completo con visualizaciones"""
        print("🎯 INICIANDO ANÁLISIS COMPLETO DEL ALGORITMO APRIORI")
        print("="*60)
        
        # Generar dataset
        self.generate_large_dataset(n_transactions=5000, n_items=50)
        
        # Visualizar dataset
        self.visualize_dataset_overview()
        
        # Ejecutar algoritmo Apriori
        self.apriori_algorithm()
        
        # Visualizar itemsets frecuentes
        self.visualize_frequent_itemsets()
        
        # Generar reglas de asociación
        self.generate_association_rules()
        
        # Visualizar reglas de asociación
        self.visualize_association_rules()
        
        # Crear visualización de red
        self.create_network_visualization()
        
        print("\n🎉 ANÁLISIS COMPLETADO")
        print("="*60)
        print(f"📊 Transacciones analizadas: {len(self.transactions)}")
        print(f"🔍 Itemsets frecuentes encontrados: {sum(len(itemsets) for itemsets in self.frequent_itemsets.values())}")
        print(f"📋 Reglas de asociación generadas: {len(self.association_rules)}")
        
        return {
            'transactions': len(self.transactions),
            'frequent_itemsets': self.frequent_itemsets,
            'association_rules': self.association_rules
        }

# Ejecutar el análisis
if __name__ == "__main__":
    # Crear instancia del visualizador
    apriori_viz = AprioriVisualizer(min_support=0.02, min_confidence=0.6)
    
    # Ejecutar análisis completo
    results = apriori_viz.run_complete_analysis()
    
    print("\n" + "="*60)
    print("🎓 EXPLICACIÓN DEL ALGORITMO APRIORI")
    print("="*60)
    print("""
    El algoritmo Apriori funciona en los siguientes pasos:
    
    1. 📊 GENERACIÓN DE ITEMSETS FRECUENTES:
       - Comienza con itemsets de tamaño 1
       - Calcula el soporte de cada itemset
       - Filtra aquellos que superan el soporte mínimo
    
    2. 🔄 ITERACIÓN:
       - Genera candidatos de tamaño k+1 combinando itemsets de tamaño k
       - Aplica la propiedad Apriori: si un itemset no es frecuente,
         ningún superconjunto puede ser frecuente
    
    3. 📋 GENERACIÓN DE REGLAS:
       - A partir de itemsets frecuentes, genera reglas A → B
       - Calcula métricas: soporte, confianza y lift
       - Filtra reglas que superan la confianza mínima
    
    4. 📈 MÉTRICAS IMPORTANTES:
       - Soporte: P(A ∪ B) - Frecuencia del itemset completo
       - Confianza: P(B|A) - Probabilidad de B dado A
       - Lift: P(B|A)/P(B) - Mejora sobre probabilidad base
    """)
