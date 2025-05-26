import pandas as pd

df = pd.read_csv("citation_edges.csv")

# Remove exact duplicate rows
df = df.drop_duplicates()

# Optional: save again
df.to_csv("citation_edges_deduplicated.csv", index=False)
