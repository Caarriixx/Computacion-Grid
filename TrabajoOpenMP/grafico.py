import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Cargar los datos desde el archivo CSV
df = pd.read_csv("resultados.csv")

# Número de hilos
hilos = df["Hilos"]

# Crear el gráfico de barras
bar_width = 0.15  # Ancho de las barras
x = np.arange(len(hilos))  # Posiciones en el eje X

# Crear la figura y los ejes
plt.figure(figsize=(10, 6))

# Graficar cada versión
for i, columna in enumerate(df.columns[1:]):
    plt.bar(x + i * bar_width, df[columna], width=bar_width, label=columna)

# Configuración del gráfico
plt.xlabel("Número de hilos")
plt.ylabel("Tiempo de ejecución (segundos)")
plt.title("Comparación de Rendimiento de heat.c con OpenMP")
plt.xticks(x + (bar_width * 2), hilos)  # Ajuste para que las etiquetas queden centradas
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Mostrar el gráfico
plt.show()
