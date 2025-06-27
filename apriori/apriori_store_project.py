import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt

# -----------------------------
# 1. Cargar dataset generado
# -----------------------------
print("📁 Cargando dataset...")
df_raw = pd.read_csv("market_basket.csv", header=None)

transactions = df_raw.apply(lambda row: row.dropna().tolist(), axis=1).tolist()
print(f"🛒 Total de transacciones: {len(transactions)}")

# -----------------------------
# 2. Simulación paso a paso
# -----------------------------
print("\n🔄 Procesando transacciones una a una...")
processed = []
for i in tqdm(range(len(transactions)), desc="Formateando transacciones"):
    processed.append(transactions[i])

# -----------------------------
# 3. Codificar los datos
# -----------------------------
print("\n✅ Transformando datos a formato para Apriori...")
te = TransactionEncoder()
te_array = te.fit(processed).transform(processed)
df = pd.DataFrame(te_array, columns=te.columns_)

# -----------------------------
# 4. Aplicar Apriori
# -----------------------------
print("\n🚀 Ejecutando algoritmo Apriori...")
frequent_itemsets = apriori(df, min_support=0.05, use_colnames=True)
print(f"\n📌 Itemsets frecuentes encontrados: {len(frequent_itemsets)}")

# -----------------------------
# 5. Reglas de asociación
# -----------------------------
print("\n📈 Generando reglas de asociación...")
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.3)
rules = rules[rules["lift"] > 1.1]
rules = rules.sort_values(by="lift", ascending=False)

if rules.empty:
    print("⚠️ No se encontraron reglas con los parámetros dados. Ajusta los valores.")
else:
    print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))

    # -----------------------------
    # 6. Visualización en grafo
    # -----------------------------
    print("\n🔗 Visualizando reglas principales...")

    G = nx.DiGraph()
    for _, row in rules.head(10).iterrows():
        for ant in row['antecedents']:
            for cons in row['consequents']:
                G.add_edge(ant, cons, weight=row['confidence'])

    plt.figure(figsize=(10, 6))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_color='lightgreen', node_size=2000, font_size=10)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels={(k[0], k[1]): f"{v:.2f}" for k, v in edge_labels.items()}
    )
    plt.title("Reglas de Asociación - Top 10 (Apriori)")
    plt.show()
