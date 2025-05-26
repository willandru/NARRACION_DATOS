import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# === 1. Cargar el CSV completo ===
df = pd.read_csv("theory3_sentiment_metrics.csv")  # Reemplaza con tu ruta real

# === 2. Normalizar todo el dataset entre [-1, 1] ===
def normalize(series):
    min_val = series.min()
    max_val = series.max()
    return 2 * (series - min_val) / (max_val - min_val) - 1

df[["Polarity", "Subjectivity", "Readability"]] = df[["Polarity", "Subjectivity", "Readability"]].apply(normalize)

# === 3. Seleccionar teorías de interés ===
selected_theories = [
    "Relativistic quantum mechanics",
    "Big Bang",
    "Chaos theory",
    "Classical electromagnetism",
    "Quantum chromodynamics",
    "Quantum electrodynamics",
    "Quantum mechanics",
    "Standard Model",
    "Theory of relativity",
    "string theory"
]

df_sel = df[df["Theory"].isin(selected_theories)].set_index("Theory")

# === 4. Graficar ===
x = np.arange(len(df_sel))
width = 0.25
metrics = ["Polarity", "Subjectivity", "Readability"]
colors = ["blue", "red", "green"]

plt.figure(figsize=(14, 6))
for i, (metric, color) in enumerate(zip(metrics, colors)):
    plt.bar(x + i * width, df_sel[metric], width, label=metric, color=color)

plt.xticks(x + width, df_sel.index, rotation=45, ha='right')
plt.ylabel("Normalized Value [-1, 1]")
plt.title("Métricas de sentimiento según la teoría")
plt.legend()
plt.tight_layout()
plt.grid(axis='y')
plt.show()