import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class AprioriStepByStep:
    """Demostración paso a paso del algoritmo Apriori"""
    
    def __init__(self):
        self.transactions = []
        self.step_results = {}
        
    def create_simple_dataset(self):
        """Crea un dataset simple para demostración paso a paso"""
        print("📝 Creando dataset simple para demostración...")
        
        # Dataset pequeño y comprensible
        transactions = [
            ['Pan', 'Leche'],
            ['Pan', 'Mantequilla', 'Huevos'],
            ['Leche', 'Mantequilla'],
            ['Pan', 'Leche', 'Mantequilla', 'Huevos'],
            ['Pan', 'Leche'],
            ['Mantequilla', 'Huevos'],
            ['Pan', 'Mantequilla'],
            ['Leche', 'Huevos'],
            ['Pan', 'Leche', 'Mantequilla'],
            ['Pan', 'Huevos']
        ]
        
        self.transactions = transactions
        print(f"✅ Dataset creado con {len(transactions)} transacciones")
        
        # Mostrar transacciones
        print("\n📋 TRANSACCIONES:")
        for i, transaction in enumerate(transactions, 1):
            print(f"T{i}: {transaction}")
        
        return transactions
    
    def step_1_count_items(self, min_support=0.3):
        """Paso 1: Contar items individuales"""
        print(f"\n🔍 PASO 1: Contando items individuales (soporte mínimo: {min_support})")
        print("="*50)
        
        # Contar cada item
        item_counts = Counter()
        for transaction in self.transactions:
            for item in transaction:
                item_counts[item] += 1
        
        total_transactions = len(self.transactions)
        
        # Calcular soporte y filtrar
        frequent_1_itemsets = []
        print("Item\t\tFrecuencia\tSoporte\t\tFrecuente?")
        print("-" * 50)
        
        for item, count in sorted(item_counts.items()):
            support = count / total_transactions
            is_frequent = support >= min_support
            status = "✅ SÍ" if is_frequent else "❌ NO"
            
            print(f"{item:<12}\t{count}\t\t{support:.2f}\t\t{status}")
            
            if is_frequent:
                frequent_1_itemsets.append(([item], support))
        
        self.step_results[1] = frequent_1_itemsets
        print(f"\n✅ Itemsets frecuentes de tamaño 1: {len(frequent_1_itemsets)}")
        
        return frequent_1_itemsets
    
    def step_2_generate_pairs(self, frequent_1_itemsets, min_support=0.3):
        """Paso 2: Generar y evaluar pares"""
        print(f"\n🔍 PASO 2: Generando pares de items (soporte mínimo: {min_support})")
        print("="*50)
        
        # Extraer items frecuentes
        frequent_items = [itemset[0][0] for itemset in frequent_1_itemsets]
        print(f"Items frecuentes: {frequent_items}")
        
        # Generar todos los pares posibles
        from itertools import combinations
        candidate_pairs = list(combinations(frequent_items, 2))
        
        print(f"\nCandidatos generados: {len(candidate_pairs)}")
        for pair in candidate_pairs:
            print(f"  {list(pair)}")
        
        # Evaluar soporte de cada par
        frequent_2_itemsets = []
        total_transactions = len(self.transactions)
        
        print(f"\nEvaluando soporte de pares:")
        print("Par\t\t\tFrecuencia\tSoporte\t\tFrecuente?")
        print("-" * 60)
        
        for pair in candidate_pairs:
            count = 0
            for transaction in self.transactions:
                if set(pair).issubset(set(transaction)):
                    count += 1
            
            support = count / total_transactions
            is_frequent = support >= min_support
            status = "✅ SÍ" if is_frequent else "❌ NO"
            
            pair_str = f"{list(pair)}"
            print(f"{pair_str:<15}\t{count}\t\t{support:.2f}\t\t{status}")
            
            if is_frequent:
                frequent_2_itemsets.append((list(pair), support))
        
        self.step_results[2] = frequent_2_itemsets
        print(f"\n✅ Itemsets frecuentes de tamaño 2: {len(frequent_2_itemsets)}")
        
        return frequent_2_itemsets
    
    def step_3_generate_triplets(self, frequent_2_itemsets, min_support=0.3):
        """Paso 3: Generar y evaluar tripletas"""
        print(f"\n🔍 PASO 3: Generando tripletas de items (soporte mínimo: {min_support})")
        print("="*50)
        
        if len(frequent_2_itemsets) < 2:
            print("❌ No hay suficientes pares frecuentes para generar tripletas")
            return []
        
        # Generar candidatos de tamaño 3
        candidates = []
        n = len(frequent_2_itemsets)
        
        for i in range(n):
            for j in range(i + 1, n):
                itemset1 = set(frequent_2_itemsets[i][0])
                itemset2 = set(frequent_2_itemsets[j][0])
                union = itemset1.union(itemset2)
                
                if len(union) == 3:
                    candidates.append(sorted(list(union)))
        
        # Remover duplicados
        unique_candidates = []
        for candidate in candidates:
            if candidate not in unique_candidates:
                unique_candidates.append(candidate)
        
        print(f"Candidatos generados: {len(unique_candidates)}")
        for candidate in unique_candidates:
            print(f"  {candidate}")
        
        # Evaluar soporte
        frequent_3_itemsets = []
        total_transactions = len(self.transactions)
        
        if unique_candidates:
            print(f"\nEvaluando soporte de tripletas:")
            print("Tripleta\t\t\tFrecuencia\tSoporte\t\tFrecuente?")
            print("-" * 70)
            
            for candidate in unique_candidates:
                count = 0
                for transaction in self.transactions:
                    if set(candidate).issubset(set(transaction)):
                        count += 1
                
                support = count / total_transactions
                is_frequent = support >= min_support
                status = "✅ SÍ" if is_frequent else "❌ NO"
                
                candidate_str = f"{candidate}"
                print(f"{candidate_str:<20}\t{count}\t\t{support:.2f}\t\t{status}")
                
                if is_frequent:
                    frequent_3_itemsets.append((candidate, support))
        
        self.step_results[3] = frequent_3_itemsets
        print(f"\n✅ Itemsets frecuentes de tamaño 3: {len(frequent_3_itemsets)}")
        
        return frequent_3_itemsets
    
    def generate_association_rules_demo(self, min_confidence=0.6):
        """Genera reglas de asociación con explicación detallada"""
        print(f"\n📋 GENERACIÓN DE REGLAS DE ASOCIACIÓN (confianza mínima: {min_confidence})")
        print("="*70)
        
        rules = []
        
        # Procesar itemsets de tamaño 2 y 3
        for k in [2, 3]:
            if k not in self.step_results or not self.step_results[k]:
                continue
            
            print(f"\n🔍 Procesando itemsets de tamaño {k}:")
            
            for itemset, support in self.step_results[k]:
                print(f"\nItemset: {itemset} (soporte: {support:.2f})")
                
                # Generar todas las reglas posibles
                from itertools import combinations
                
                for i in range(1, len(itemset)):
                    for antecedent in combinations(itemset, i):
                        antecedent_list = list(antecedent)
                        consequent_list = [item for item in itemset if item not in antecedent_list]
                        
                        # Calcular soporte del antecedente
                        antecedent_count = 0
                        for transaction in self.transactions:
                            if set(antecedent_list).issubset(set(transaction)):
                                antecedent_count += 1
                        
                        antecedent_support = antecedent_count / len(self.transactions)
                        
                        # Calcular confianza
                        confidence = support / antecedent_support if antecedent_support > 0 else 0
                        
                        # Calcular lift
                        consequent_count = 0
                        for transaction in self.transactions:
                            if set(consequent_list).issubset(set(transaction)):
                                consequent_count += 1
                        
                        consequent_support = consequent_count / len(self.transactions)
                        lift = confidence / consequent_support if consequent_support > 0 else 0
                        
                        rule_str = f"{antecedent_list} → {consequent_list}"
                        print(f"  Regla: {rule_str}")
                        print(f"    Soporte antecedente: {antecedent_support:.2f}")
                        print(f"    Confianza: {support:.2f} / {antecedent_support:.2f} = {confidence:.2f}")
                        print(f"    Lift: {lift:.2f}")
                        
                        if confidence >= min_confidence:
                            print(f"    ✅ REGLA VÁLIDA (confianza >= {min_confidence})")
                            rules.append({
                                'antecedent': antecedent_list,
                                'consequent': consequent_list,
                                'support': support,
                                'confidence': confidence,
                                'lift': lift
                            })
                        else:
                            print(f"    ❌ Regla rechazada (confianza < {min_confidence})")
        
        print(f"\n🎉 REGLAS DE ASOCIACIÓN FINALES: {len(rules)}")
        print("="*50)
        
        for i, rule in enumerate(sorted(rules, key=lambda x: x['confidence'], reverse=True), 1):
            print(f"{i}. {rule['antecedent']} → {rule['consequent']}")
            print(f"   Soporte: {rule['support']:.2f} | Confianza: {rule['confidence']:.2f} | Lift: {rule['lift']:.2f}")
        
        return rules
    
    def visualize_step_by_step(self):
        """Crea visualizaciones paso a paso"""
        print("\n📊 Creando visualizaciones paso a paso...")
        
        # Crear subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Paso 1: Items Individuales', 
                          'Paso 2: Pares de Items',
                          'Paso 3: Tripletas', 
                          'Resumen del Proceso'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "table"}]]
        )
        
        # Paso 1: Items individuales
        if 1 in self.step_results:
            items_1 = [itemset[0][0] for itemset, _ in self.step_results[1]]
            supports_1 = [support for _, support in self.step_results[1]]
            
            fig.add_trace(
                go.Bar(x=items_1, y=supports_1, name="Items Frecuentes",
                       text=[f"{s:.2f}" for s in supports_1], textposition='auto'),
                row=1, col=1
            )
        
        # Paso 2: Pares
        if 2 in self.step_results:
            pairs_2 = [' + '.join(itemset) for itemset, _ in self.step_results[2]]
            supports_2 = [support for _, support in self.step_results[2]]
            
            fig.add_trace(
                go.Bar(x=pairs_2, y=supports_2, name="Pares Frecuentes",
                       text=[f"{s:.2f}" for s in supports_2], textposition='auto'),
                row=1, col=2
            )
        
        # Paso 3: Tripletas
        if 3 in self.step_results:
            triplets_3 = [' + '.join(itemset) for itemset, _ in self.step_results[3]]
            supports_3 = [support for _, support in self.step_results[3]]
            
            fig.add_trace(
                go.Bar(x=triplets_3, y=supports_3, name="Tripletas Frecuentes",
                       text=[f"{s:.2f}" for s in supports_3], textposition='auto'),
                row=2, col=1
            )
        
        # Resumen
        summary_data = [
            ["Paso", "Descripción", "Itemsets Encontrados"],
            ["1", "Items individuales", len(self.step_results.get(1, []))],
            ["2", "Pares de items", len(self.step_results.get(2, []))],
            ["3", "Tripletas de items", len(self.step_results.get(3, []))]
        ]
        
        fig.add_trace(
            go.Table(
                header=dict(values=summary_data[0]),
                cells=dict(values=[[row[i] for row in summary_data[1:]] for i in range(3)])
            ),
            row=2, col=2
        )
        
        fig.update_layout(height=800, title_text="Algoritmo Apriori - Demostración Paso a Paso")
        fig.show()
    
    def run_step_by_step_demo(self):
        """Ejecuta la demostración completa paso a paso"""
        print("🎯 DEMOSTRACIÓN PASO A PASO DEL ALGORITMO APRIORI")
        print("="*60)
        
        # Crear dataset simple
        self.create_simple_dataset()
        
        # Ejecutar pasos
        frequent_1 = self.step_1_count_items(min_support=0.3)
        frequent_2 = self.step_2_generate_pairs(frequent_1, min_support=0.3)
        frequent_3 = self.step_3_generate_triplets(frequent_2, min_support=0.3)
        
        # Generar reglas
        rules = self.generate_association_rules_demo(min_confidence=0.6)
        
        # Crear visualizaciones
        self.visualize_step_by_step()
        
        print("\n🎓 CONCEPTOS CLAVE DEMOSTRADOS:")
        print("="*40)
        print("✅ Principio Apriori: Si un itemset no es frecuente, ningún superconjunto puede serlo")
        print("✅ Generación de candidatos: Combinación sistemática de itemsets frecuentes")
        print("✅ Poda: Eliminación de candidatos que no cumplen soporte mínimo")
        print("✅ Reglas de asociación: Extracción de patrones A → B con métricas")
        
        return {
            'frequent_itemsets': self.step_results,
            'association_rules': rules
        }

# Ejecutar demostración paso a paso
if __name__ == "__main__":
    demo = AprioriStepByStep()
    results = demo.run_step_by_step_demo()
