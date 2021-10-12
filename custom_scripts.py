import networkx as nx
from pathlib import Path
from dominant import load_graph, dominant
import random
import matplotlib.pyplot as plt

INPUT_DIR = Path(
    r"C:\Users\thiba\OneDrive - CentraleSupelec\3A_Centrale_Supélec\COURS\ALGO-ALGORITHMIQUE_AVANCEE\concours_algo\public_data\public_dataset"
)
GRAPH_FILENAME = "graph_50_50"
GRAPH_FILENAMES_LIST = [
    "graph_100_100",
    "graph_100_1000",
    "graph_100_250",
    "graph_100_500",
    "graph_250_1000",
    "graph_250_250",
    "graph_250_500",
    "graph_500_1000",
    "graph_500_500",
    "graph_50_1000",
    "graph_50_250",
    "graph_50_50",
    "graph_50_500",
]


def plot_graph(
    input_dir: Path,
    graph_filename: str,
):
    graph = load_graph(input_dir / graph_filename)
    plt.title(graph_filename)
    labels = {n: str(n) + "\n\n" + str(graph.nodes[n]["weight"]) for n in graph.nodes}
    nx.draw(graph, with_labels=True, labels=labels, font_weight="bold")


def plot_random_graph(
    input_dir: Path,
    graph_filenames_list: list,
):
    graph_filename = random.choice(graph_filenames_list)
    plt.title(graph_filename)
    plot_graph(input_dir=input_dir, graph_filename=graph_filename)


def compute_graph_dominant_set_score(
        input_dir: Path,
        graph_filename: str,
        dominant_set: list,
) -> int:
    graph = load_graph(input_dir / graph_filename)
    return sum(
        [
            weight_dict["weight"]
            for node_number, weight_dict in dict(graph.nodes(data=True)).items()
            if node_number in dominant_set
        ]
    )


def compute_global_score(
        input_dir: Path,
        graph_filenames_list: list,
) -> int:
    total_score = 0
    for graph_filename in graph_filenames_list:
        graph = load_graph(input_dir / graph_filename)
        dominating_set = dominant(graph)
        total_score += compute_graph_dominant_set_score(
            input_dir=input_dir,
            graph_filename=graph_filename,
            dominant_set=dominating_set,
        )
    return total_score


def get_node_weight(node: tuple):
    return node[1]["weight"]


def test_dominant(
    input_dir: Path,
    graph_filename: str,
):
    graph = load_graph(input_dir / graph_filename)
    return dominant(graph)




# -----
# DEBUG

# plot_graph(INPUT_DIR, GRAPH_FILENAME)
# plot_random_graph(INPUT_DIR, GRAPH_FILENAMES_LIST)
# get_node_neighbors(INPUT_DIR, GRAPH_FILENAME, 6)
# test_dominant(INPUT_DIR, GRAPH_FILENAME)
# compute_global_score(INPUT_DIR, GRAPH_FILENAMES_LIST)