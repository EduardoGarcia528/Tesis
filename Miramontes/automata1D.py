import numpy as np
import matplotlib.pyplot as plt

def regla_a_tabla(rule_number):
    bin_str = f"{rule_number:08b}"  # ej. 30 -> '00011110'

    patrones = ["111", "110", "101", "100", "011", "010", "001", "000"]

    tabla = {pat: int(bit) for pat, bit in zip(patrones, bin_str)}
    return tabla

def paso_automata_1D(estado, tabla_regla):
    N = len(estado)
    nuevo = np.zeros_like(estado)

    for i in range(N):
        izquierda = estado[(i - 1) % N]
        centro    = estado[i]
        derecha   = estado[(i + 1) % N]

        vecindad = f"{izquierda}{centro}{derecha}"
        nuevo[i] = tabla_regla[vecindad]

    return nuevo


def estado_inicial(ancho=100, modo="semilla"):
    if modo == "semilla":
        estado = np.zeros(ancho, dtype=int)
        estado[ancho // 2] = 1

    elif modo == "periodica":
        patron = np.array([1, 0, 0, 0, 0], dtype=int)
        repeticiones = ancho // 5 + 1
        estado = np.tile(patron, repeticiones)[:ancho]

    elif modo == "aleatoria":
        estado = np.random.randint(0, 2, size=ancho, dtype=int)
    return estado

def correr_automata(rule_number, pasos = 100,
                    ancho = 100,
                    modo_ini = "semilla"):
    tabla = regla_a_tabla(rule_number)
    estado = estado_inicial(ancho=ancho, modo=modo_ini)

    historia = np.zeros((pasos, ancho), dtype=int)

    for t in range(pasos):
        historia[t] = estado
        estado = paso_automata_1D(estado, tabla)

    return historia

def graficar_automata(historia, rule_number, modo_ini):

    plt.figure(figsize=(6, 6))
    plt.imshow(historia, cmap="binary", interpolation="nearest", aspect="auto")
    plt.xlabel("Celda")
    plt.ylabel("Tiempo")
    plt.title(f"Autómata 1D - Regla {rule_number} - Condición {modo_ini}")
    plt.tight_layout()
    plt.show()

def entropy_average(historia):
    pasos, ancho = historia.shape
    entropias = []
    averages = []

    for t in range(pasos):
        estado = historia[t]
        p1 = np.mean(estado)
        p0 = 1 - p1

        if p1 in [0, 1]:
            entropia = 0
        else:
            entropia = - (p0 * np.log2(p0) + p1 * np.log2(p1))

        entropias.append(entropia)
        inicio = max(0, t - 9)
        ventana = entropias[inicio:t + 1]
        averages.append(np.mean(ventana))

    return entropias, averages

if __name__ == "__main__":
    for regla in [18, 22, 30, 45, 110]:
        for inicial in ["semilla", "periodica", "aleatoria"]:
            ancho = 100
            pasos = 100

            historia = correr_automata(rule_number=regla,
                                    pasos=pasos,
                                    ancho=ancho,
                                    modo_ini=inicial)
            graficar_automata(historia, regla, modo_ini=inicial)
            entropias, averages = entropy_average(historia)
            plt.figure()
            plt.plot(entropias, label="Entropía")
            plt.plot(averages, label="Promedio de 1's")
            plt.xlabel("Tiempo")
            plt.ylabel("Valor")
            plt.title(f"Entropía y Promedio - Regla {regla} - Condición {inicial}")
            plt.legend()
            plt.tight_layout()
            # plt.savefig(r"C:/Users/PC BULLOCK/Downloads/automatas/" + f"regla_{regla}_{inicial}_H.png")
            plt.show()