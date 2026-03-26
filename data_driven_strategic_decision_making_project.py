import pandas as pd
import networkx as nx

ro = pd.read_csv("data/RO_edges.csv")

# Graph
G_ro = nx.from_pandas_edgelist(ro, source="node_1", target="node_2")

# Nodes and  edges 
print("Romania:")
print("Nodes:", G_ro.number_of_nodes())
print("Edges:", G_ro.number_of_edges())

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

# Romania degree distribution
degrees_ro = [d for _, d in G_ro.degree()]

plt.figure(figsize=(8, 5))
plt.hist(degrees_ro, bins=50)
plt.title("Degree Distribution - Romania")
plt.xlabel("Degree")
plt.ylabel("Frequency")
plt.savefig("degree_distribution_romania.png", dpi=300, bbox_inches="tight")
print("Figure saved as degree_distribution_romania.png")

# Connected components - Romania
components_ro = list(nx.connected_components(G_ro))

print("Romania connected components:", len(components_ro))
print("Largest component size:", len(max(components_ro, key=len)))

# Largest connected component subgraph
largest_cc_ro = G_ro.subgraph(max(components_ro, key=len)).copy()

print("Largest CC nodes:", largest_cc_ro.number_of_nodes())
print("Largest CC edges:", largest_cc_ro.number_of_edges())

import random

sample_nodes = random.sample(list(G_ro.nodes()), 100)
path_lengths = []

for source in sample_nodes:
    lengths = nx.single_source_shortest_path_length(G_ro, source)
    path_lengths.extend(lengths.values())

approx_avg_path = sum(path_lengths) / len(path_lengths)

print("Approximate average shortest path length (Romania):", approx_avg_path)
print("Romania average clustering coefficient:", nx.average_clustering(G_ro))
print("Romania density:", nx.density(G_ro))

# Centrality measures - Romania
degree_centrality_ro = nx.degree_centrality(G_ro)
betweenness_centrality_ro = nx.betweenness_centrality(G_ro, k=500, seed=42)
pagerank_ro = nx.pagerank(G_ro)
eigenvector_ro = nx.eigenvector_centrality(G_ro, max_iter=1000)

def top_n(metric_dict, n=10):
    return sorted(metric_dict.items(), key=lambda x: x[1], reverse=True)[:n]

print("Top 10 Degree Centrality:", top_n(degree_centrality_ro))
print("\nTop 10 Betweenness Centrality:", top_n(betweenness_centrality_ro))
print("\nTop 10 PageRank:", top_n(pagerank_ro))
print("\nTop 10 Eigenvector Centrality:", top_n(eigenvector_ro))

# Synthetic graphs
n = G_ro.number_of_nodes()
m = G_ro.number_of_edges()

# ER graph 
p = (2 * m) / (n * (n - 1))
G_er = nx.erdos_renyi_graph(n, p, seed=42)

# BA graph 
m_ba = max(1, round(m / n))
G_ba = nx.barabasi_albert_graph(n, m_ba, seed=42)

# WS graph 
k_ws = round((2 * m) / n)
if k_ws % 2 != 0:
    k_ws += 1

G_ws = nx.watts_strogatz_graph(n, k_ws, 0.1, seed=42)

print("ER graph:", G_er.number_of_nodes(), G_er.number_of_edges())
print("BA graph:", G_ba.number_of_nodes(), G_ba.number_of_edges())
print("WS graph:", G_ws.number_of_nodes(), G_ws.number_of_edges())

def graph_summary(G, name):
    components = list(nx.connected_components(G))
    largest_cc = G.subgraph(max(components, key=len)).copy()

    sample_nodes = random.sample(list(largest_cc.nodes()), min(100, largest_cc.number_of_nodes()))
    path_lengths = []

    for source in sample_nodes:
        lengths = nx.single_source_shortest_path_length(largest_cc, source)
        path_lengths.extend(lengths.values())

    approx_avg_path = sum(path_lengths) / len(path_lengths)

    print(f"\n{name}")
    print("Nodes:", G.number_of_nodes())
    print("Edges:", G.number_of_edges())
    print("Connected components:", len(components))
    print("Largest component size:", largest_cc.number_of_nodes())
    print("Approx. average shortest path length:", approx_avg_path)
    print("Average clustering coefficient:", nx.average_clustering(G))
    print("Density:", nx.density(G))

    
graph_summary(G_ro, "Romania Real Graph")
graph_summary(G_er, "ER Graph")
graph_summary(G_ba, "BA Graph")
graph_summary(G_ws, "WS Graph")

# Linear Threshold Model

import random

random.seed(42)
thresholds = {node: random.random() for node in G_ro.nodes()}

def linear_threshold_model(G, seed_nodes, thresholds):
    newly_active_nodes = set(seed_nodes)
    all_activated_nodes = set(seed_nodes)

    while newly_active_nodes:
        current_iteration_activated_nodes = set()

        for u in G.nodes():
            if u not in all_activated_nodes:
                if G.degree(u) > 0:
                    active_neighbors = [v for v in G.neighbors(u) if v in all_activated_nodes]
                    influence_sum = len(active_neighbors) / G.degree(u)
                else:
                    influence_sum = 0

                if influence_sum >= thresholds[u]:
                    current_iteration_activated_nodes.add(u)

        newly_active_nodes = current_iteration_activated_nodes
        all_activated_nodes.update(newly_active_nodes)

    return all_activated_nodes


method_degree = [node for node, _ in sorted(degree_centrality_ro.items(), key=lambda x: x[1], reverse=True)[:5]]
method_betweenness = [node for node, _ in sorted(betweenness_centrality_ro.items(), key=lambda x: x[1], reverse=True)[:5]]
method_pagerank = [node for node, _ in sorted(pagerank_ro.items(), key=lambda x: x[1], reverse=True)[:5]]
method_eigenvector = [node for node, _ in sorted(eigenvector_ro.items(), key=lambda x: x[1], reverse=True)[:5]]

activated_degree = linear_threshold_model(G_ro, method_degree, thresholds)
activated_betweenness = linear_threshold_model(G_ro, method_betweenness, thresholds)
activated_pagerank = linear_threshold_model(G_ro, method_pagerank, thresholds)
activated_eigenvector = linear_threshold_model(G_ro, method_eigenvector, thresholds)

results = {
    "Degree": len(activated_degree),
    "Betweenness": len(activated_betweenness),
    "PageRank": len(activated_pagerank),
    "Eigenvector": len(activated_eigenvector)
}

print("\nLinear threshold results comparison:")
for method, spread in results.items():
    print(f"{method}: {spread}")

best_method = max(results, key=results.get)
print("\nBest method under LT model:", best_method)

# greedy algorithm

def greedy_seed_selection(G, candidate_nodes, k, thresholds):
    selected_seeds = []
    
    for step in range(k):
        best_node = None
        best_spread = -1
        
        for node in candidate_nodes:
            if node not in selected_seeds:
                trial_seeds = selected_seeds + [node]
                activated_nodes = linear_threshold_model(G, trial_seeds, thresholds)
                spread = len(activated_nodes)
                
                if spread > best_spread:
                    best_spread = spread
                    best_node = node
        
        selected_seeds.append(best_node)
        print(f"Step {step+1}: selected node {best_node}, spread = {best_spread}")
    
    return selected_seeds

top_candidates = list(set(
    [node for node, _ in sorted(degree_centrality_ro.items(), key=lambda x: x[1], reverse=True)[:10]] +
    [node for node, _ in sorted(betweenness_centrality_ro.items(), key=lambda x: x[1], reverse=True)[:10]] +
    [node for node, _ in sorted(pagerank_ro.items(), key=lambda x: x[1], reverse=True)[:10]] +
    [node for node, _ in sorted(eigenvector_ro.items(), key=lambda x: x[1], reverse=True)[:10]]
))

greedy_seeds = greedy_seed_selection(G_ro, top_candidates, 2, thresholds)
greedy_activated = linear_threshold_model(G_ro, greedy_seeds, thresholds)

print("\nGreedy algorithm results:", greedy_seeds)
print("Total activated nodes by greedy algorithm:", len(greedy_activated))