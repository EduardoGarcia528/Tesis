import pandas as pd

df = pd.read_csv("melodies_found.csv")

df.head()       # primeras 5 filas


print(df.shape)        # (n_filas, n_columnas)
print(df.columns)      # nombres de columnas
print(df.info())       # tipos de datos y valores no nulos
print(df.describe())   # estadísticas de columnas numéricas

print(df["composer"].unique())