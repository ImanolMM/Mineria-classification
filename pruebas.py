import pandas as pd
from tensorflow.python.keras.utils.all_utils

# Carga del dataset
df = pd.read_csv("/home/imanol/WIP/Mineria/data/dev.csv")

# Eliminar valores NaN en la columna objetivo (gs_text34)
df = df.dropna(subset=["gs_text34"])

# Contar clases únicas
unique_classes = df["gs_text34"].nunique()

print(f"Número de clases únicas: {unique_classes}")
