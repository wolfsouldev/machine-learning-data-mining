import random
import csv

products = [
    "pan", "leche", "queso", "mantequilla", "café", "té", "azúcar", "harina",
    "huevos", "pollo", "carne", "pescado", "arroz", "pasta", "aceite",
    "tomate", "cebolla", "papa", "zanahoria", "manzana", "banana", "naranja",
    "uvas", "yogurt", "galletas", "refresco", "agua", "chocolate", "jamón", "sal"
]

NUM_TRANSACTIONS = 5000

with open("market_basket.csv", "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    for _ in range(NUM_TRANSACTIONS):
        num_items = random.randint(1, 8)
        transaction = random.sample(products, num_items)
        writer.writerow(transaction)

print("✅ Dataset 'market_basket.csv' generado con éxito.")
