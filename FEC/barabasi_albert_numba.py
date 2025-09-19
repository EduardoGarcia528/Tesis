import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.stats import linregress
from numba import njit

import numpy as np
from numba import njit
from numba.typed import List

@njit
def weighted_choice_no_replace(probs, m):
    """
    Selecciona m índices distintos según probabilidades dadas.
    Equivalente a np.random.choice(len(probs), size=m, replace=False, p=probs)
    pero compatible con Numba.
    """
    chosen = np.empty(m, dtype=np.int64)
    available = np.arange(len(probs))
    weights = probs.copy()

    for k in range(m):
        weights_sum = np.sum(weights)
        r = np.random.random() * weights_sum

        cum_sum = 0.0
        for idx in range(len(weights)):
            cum_sum += weights[idx]
            if cum_sum >= r:
                chosen[k] = available[idx]

                # "eliminar" ese índice de disponibles
                weights[idx] = 0.0
                break
    return chosen


def barabasi_albert_inicial(m0, seed=None):
    if seed is not None:
        np.random.seed(seed)

    # typed.List para ser usable en Numba
    adj_list = List()
    for _ in range(m0):
        adj_list.append(List.empty_list(np.int64))

    edges = List()
    for i in range(m0):
        for j in range(i+1, m0):
            adj_list[i].append(j)
            adj_list[j].append(i)
            edges.append(np.array([i, j], dtype=np.int64))
    return adj_list, edges


@njit
def barabasi_albert_process(N, m0, m, adj_list, edges):
    for new_node in range(m0, N):
        # calcular grados
        degrees = np.empty(len(adj_list), dtype=np.int64)
        for i in range(len(adj_list)):
            degrees[i] = len(adj_list[i])
        probs = degrees / np.sum(degrees)

        chosen = weighted_choice_no_replace(probs, m)

        # añadir nuevo nodo a adj_list
        new_neighbors = List.empty_list(np.int64)
        adj_list.append(new_neighbors)

        for c in chosen:
            adj_list[new_node].append(c)
            adj_list[c].append(new_node)
            edges.append(np.array([min(c, new_node), max(c, new_node)], dtype=np.int64))

    return adj_list, edges


def Graficar_red_inicial(edges, m0, plot=True):
    if plot:
        G_init = nx.Graph()
        G_init.add_edges_from(edges)
        degrees_init = dict(G_init.degree())
        node_sizes_init = [degrees_init[n] * 500 for n in G_init.nodes()]

        plt.figure(figsize=(5, 5))
        nx.draw(G_init, node_size=node_sizes_init, labels={n: n for n in G_init.nodes()})
        plt.title(f"Red inicial (m0={m0})")
        plt.show()



def barabasi_albert_final_graph(N, m0, m, edges, plot=True):
    G_final = nx.Graph()
    G_final.add_edges_from(edges)
    degrees_final = dict(G_final.degree())
    node_sizes_final = [degrees_final[n] * 50 for n in G_final.nodes()]

    # Seleccionar los 5 nodos más conectados
    top5_nodes = sorted(degrees_final, key=degrees_final.get, reverse=True)[:5]
    labels_top5 = {n: n for n in top5_nodes}
    node_colors = ["red" if n in top5_nodes else "lightgray" for n in G_final.nodes()]

    if plot:
        # --- Graficar red final ---
        plt.figure(figsize=(6, 6))
        nx.draw(
            G_final,
            node_size=node_sizes_final,
            node_color=node_colors,
            labels=labels_top5,
            font_size=8,
            font_color="black"
        )
        plt.title(f"Red final Barabási–Albert (N={N}, m0={m0}, m={m})")
        plt.show()

        # --- Graficar distribución de grado ---
        degrees = np.array(list(degrees_final.values()))
        unique, counts = np.unique(degrees, return_counts=True)

        plt.figure(figsize=(12, 4))

        # Histograma normal
        plt.subplot(1, 2, 1)
        plt.bar(unique, counts, width=0.8, color="steelblue")
        plt.xlabel("Grado k")
        plt.ylabel("Número de nodos")
        plt.title("Distribución de grados")

        # Escala log-log y ajuste de ley de potencia
        plt.subplot(1, 2, 2)
        plt.scatter(unique, counts, color="darkred", label="Datos")

        # Ajuste lineal en log-log (ignorando k pequeños)
        mask = unique < 10  # filtra grados pequeños
        if np.sum(mask) >= 2:  # al menos dos puntos para ajustar
            log_k = np.log(unique[mask])
            log_counts = np.log(counts[mask])
            slope, intercept, r, p, se = linregress(log_k, log_counts)
            gamma = -slope

            # Curva ajustada
            fit_y = np.exp(intercept) * unique[mask] ** slope
            plt.plot(unique[mask], fit_y, "--", color="blue",
                     label=f"Ajuste: γ ≈ {gamma:.2f}")

        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Grado k (log)")
        plt.ylabel("Número de nodos (log)")
        plt.title("Distribución de grados (log-log)")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return edges, degrees_final

def main(N,m0,m):
    adj_list, edges = barabasi_albert_inicial(m0)# Ejemplo de uso
    print("inicial")
    Graficar_red_inicial(edges, m0)
    print("graficada")
    adj_list_final, edges = barabasi_albert_process(N, m0,m, adj_list, edges, plot=True)
    print("proceso terminado")
    edges, degrees_final = barabasi_albert_final_graph(N, m0, m, edges, plot=True)
    return edges, adj_list_final, degrees_final


edges, adj, degrees = main(N=100_000, m0=5, m=1)
