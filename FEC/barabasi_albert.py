import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.stats import linregress

def barabasi_albert_graph(N=50, m0=5, m=2, seed=None, plot=True):

    if seed is not None:
        np.random.seed(seed)

    adj_list = [[] for _ in range(m0)]
    edges = []
    print("1")
    for i in range(m0):
        for j in range(i+1, m0):
            adj_list[i].append(j)
            adj_list[j].append(i)
            edges.append((i, j))
    print("2")
    # Graficar red inicial
    if plot:
        G_init = nx.Graph()
        G_init.add_edges_from(edges)
        degrees_init = dict(G_init.degree())
        node_sizes_init = [degrees_init[n] * 500 for n in G_init.nodes()]

        plt.figure(figsize=(5, 5))
        nx.draw(G_init, node_size=node_sizes_init, labels={n: n for n in G_init.nodes()})
        plt.title(f"Red inicial (m0={m0})")
        plt.show()
    
    # Paso 2: añadir nodos uno por uno
    for new_node in range(m0, N):
        print(new_node)
        degrees = np.array([len(neigh) for neigh in adj_list])
        probs = degrees / degrees.sum()
        chosen = np.random.choice(len(adj_list), size=m, replace=False, p=probs)

        adj_list.append([])
        for c in chosen:
            adj_list[new_node].append(c)
            adj_list[c].append(new_node)
            edges.append((min(c, new_node), max(c, new_node)))

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
    print("3")
    # --- Graficar distribución de grado ---
    degrees = np.array(list(degrees_final.values()))
    unique, counts = np.unique(degrees, return_counts=True)

    plt.figure(figsize=(12, 4))
    print("4")
    # Histograma normal
    plt.subplot(1, 2, 1)
    plt.bar(unique, counts, width=0.8, color="steelblue")
    plt.xlabel("Grado k")
    plt.ylabel("Número de nodos")
    plt.title("Distribución de grados")

    # Escala log-log y ajuste de ley de potencia
    plt.subplot(1, 2, 2)
    plt.scatter(unique, counts, color="darkred", label="Datos")

    print("5")
    # Ajuste lineal en log-log (ignorando k pequeños)
    mask = unique < 50  # filtra grados pequeños
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

    return edges, adj_list, degrees_final


# Ejemplo de uso
edges, adj, degrees = barabasi_albert_graph(N=100_000, m0=5, m=1, seed=42,plot=False)
