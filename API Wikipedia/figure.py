import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

# Cargar el CSV con todas las métricas
df = pd.read_csv("theory3_sentiment_metrics.csv")  # Cambia a la ruta real de tu archivo

# Seleccionar teorías manualmente
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

# === 3. Normalizar entre -1 y 1 ===
def normalize(series):
    min_val = series.min()
    max_val = series.max()
    return 2 * (series - min_val) / (max_val - min_val) - 1  # Escala [0,1] → [-1,1]

df_norm = df_sel.copy()
for col in ["Polarity", "Subjectivity", "Readability"]:
    df_norm[col] = normalize(df_sel[col])

# === 4. Graficar múltiples barras ===
x = np.arange(len(df_norm))
width = 0.25
metrics = ["Polarity", "Subjectivity", "Readability"]

plt.figure(figsize=(14, 6))
for i, metric in enumerate(metrics):
    plt.bar(x + i * width, df_norm[metric], width, label=metric)

plt.xticks(x + width, df_norm.index, rotation=45, ha='right')
plt.ylabel("Normalized Value [-1, 1]")
plt.title("Normalized Sentiment Metrics per Theory")
plt.legend()
plt.tight_layout()
plt.grid(axis='y')
plt.show()