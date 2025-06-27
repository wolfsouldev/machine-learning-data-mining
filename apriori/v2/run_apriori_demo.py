"""
Script principal para ejecutar las demostraciones del algoritmo Apriori
"""

print("🚀 DEMOSTRACIÓN COMPLETA DEL ALGORITMO APRIORI")
print("="*60)
print("Este script ejecutará dos demostraciones:")
print("1. 📚 Demostración paso a paso con dataset simple")
print("2. 🔬 Análisis completo con dataset grande")
print("="*60)

# Importar las clases
from apriori_step_by_step import AprioriStepByStep
from apriori_algorithm import AprioriVisualizer

def main():
    """Función principal que ejecuta ambas demostraciones"""
    
    print("\n" + "🎯" * 20)
    print("PARTE 1: DEMOSTRACIÓN EDUCATIVA PASO A PASO")
    print("🎯" * 20)
    
    # Ejecutar demostración paso a paso
    demo = AprioriStepByStep()
    step_results = demo.run_step_by_step_demo()
    
    print("\n" + "🔬" * 20)
    print("PARTE 2: ANÁLISIS COMPLETO CON DATASET GRANDE")
    print("🔬" * 20)
    
    # Ejecutar análisis completo
    apriori_viz = AprioriVisualizer(min_support=0.015, min_confidence=0.5)
    complete_results = apriori_viz.run_complete_analysis()
    
    print("\n" + "🎉" * 20)
    print("RESUMEN FINAL")
    print("🎉" * 20)
    
    print(f"""
📊 DEMOSTRACIÓN PASO A PASO:
   - Transacciones: 10
   - Itemsets frecuentes nivel 1: {len(step_results['frequent_itemsets'].get(1, []))}
   - Itemsets frecuentes nivel 2: {len(step_results['frequent_itemsets'].get(2, []))}
   - Itemsets frecuentes nivel 3: {len(step_results['frequent_itemsets'].get(3, []))}
   - Reglas de asociación: {len(step_results['association_rules'])}

🔬 ANÁLISIS COMPLETO:
   - Transacciones: {complete_results['transactions']}
   - Total itemsets frecuentes: {sum(len(itemsets) for itemsets in complete_results['frequent_itemsets'].values())}
   - Reglas de asociación: {len(complete_results['association_rules'])}

🎓 ALGORITMO APRIORI - PUNTOS CLAVE:
   ✅ Encuentra patrones frecuentes en transacciones
   ✅ Usa el principio Apriori para podar candidatos
   ✅ Genera reglas de asociación con métricas
   ✅ Aplicable en análisis de mercado, recomendaciones, etc.
    """)
    
    print("\n🏆 ¡DEMOSTRACIÓN COMPLETADA EXITOSAMENTE!")
    print("="*60)

if __name__ == "__main__":
    main()
