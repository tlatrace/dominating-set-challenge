import networkx as nx

from dominant import (
    get_node_neighbors,
    load_graph,
    get_node_degree,
    weighted_degree_0_rule,
    weighted_degree_1_rule_1,
    weighted_degree_1_rule_2,
    weighted_degree_2_rule,
    reduce_graph,
    construct_dominating_set,
    is_node_dominated,
    dominant, get_nodes_neighbors_dict,
)
from pathlib import Path

INPUT_DIR = Path(
    r"C:\Users\thiba\OneDrive - CentraleSupelec\3A_Centrale_Supélec\COURS\ALGO-ALGORITHMIQUE_AVANCEE\concours_algo\public_data\public_dataset"
)
GRAPH_FILENAME = "graph_50_50"


def test_get_node_neighbors() -> None:
    graph = load_graph(INPUT_DIR / GRAPH_FILENAME)
    node_number = 35
    assert get_node_neighbors(graph, node_number) == {
        28,
        42,
    }, f"Wrong neighborhood : should be {{28, 42}} but is {get_node_neighbors(graph, node_number)}."


def test_get_node_degree() -> None:
    graph = load_graph(INPUT_DIR / GRAPH_FILENAME)
    node_number = 35
    assert (
        get_node_degree(graph, node_number) == 2
    ), f"Node degree should be 2 but is {get_node_degree(graph, node_number)}"


def test_weighted_degree_0_rule() -> None:
    graph = load_graph(INPUT_DIR / GRAPH_FILENAME)
    graph.add_node(50)
    nodes_neighbors_dict = get_nodes_neighbors_dict(graph=graph)
    assert weighted_degree_0_rule(graph, nodes_neighbors_dict) == {50}


def test_weighted_degree_1_rule_1() -> None:
    graph = load_graph(INPUT_DIR / GRAPH_FILENAME)
    nodes_weight_dict = dict(graph.nodes(data=True))
    nodes_neighbors_dict = get_nodes_neighbors_dict(graph=graph)
    assert weighted_degree_1_rule_1(graph, nodes_weight_dict, nodes_neighbors_dict) == (
        {9},
        {0},
    )  # use plot_graph() to check visually


def test_weighted_degree_1_rule_2() -> None:
    graph = load_graph(INPUT_DIR / GRAPH_FILENAME)
    graph.add_node(50, weight=1)
    graph.add_node(51, weight=5)
    graph.add_edge(50, 9)
    graph.add_edge(51, 9)
    nodes_weight_dict = dict(graph.nodes(data=True))
    nodes_neighbors_dict = get_nodes_neighbors_dict(graph)
    assert weighted_degree_1_rule_2(graph, nodes_weight_dict, nodes_neighbors_dict) == (
        {9},
        {0, 50, 51},
    ), f"Should be ({{9}}, {{0, 50, 51}}) but is {weighted_degree_1_rule_2(graph)}"


def test_weighted_degree_2_rule() -> None:
    graph = load_graph(INPUT_DIR / GRAPH_FILENAME)
    graph.add_node(50, weight=40)
    graph.add_edge(50, 9)
    graph.add_edge(50, 0)
    nodes_weight_dict = dict(graph.nodes(data=True))
    nodes_neighbors_dict = get_nodes_neighbors_dict(graph)
    assert weighted_degree_2_rule(graph, nodes_weight_dict, nodes_neighbors_dict) == (
        {9},
        {0, 50},
    ), f"Should be {{9}} but is {weighted_degree_2_rule(graph)}"


def test_reduce_graph() -> None:
    graph = load_graph(INPUT_DIR / GRAPH_FILENAME)
    graph.add_node(50, weight=1)
    graph.add_node(51, weight=5)
    graph.add_node(52, weight=30)
    graph.add_node(53, weight=40)
    graph.add_node(54)
    graph.add_edge(50, 9)
    graph.add_edge(51, 9)
    graph.add_edge(26, 52)
    graph.add_edge(26, 53)
    graph.add_edge(52, 53)

    nodes_weight_dict = dict(graph.nodes(data=True))
    nodes_neighbors_dict = get_nodes_neighbors_dict(graph=graph)
    nodes_to_remove, dominating_nodes, reduced_graph = reduce_graph(
        graph, nodes_weight_dict, nodes_neighbors_dict
    )
    assert (nodes_to_remove, dominating_nodes) == (
        {0, 50, 51, 52, 53},
        {9, 26, 54},
    ), f"Result should be ({{0, 50, 51, 52, 53}}, {{9, 26, 54}}) but is {reduce_graph(graph)}"


def test_is_node_linked_to_dominating_nodes() -> None:
    graph = load_graph(INPUT_DIR / GRAPH_FILENAME)
    dominating_nodes = {3, 35}
    node_number = 3
    nodes_neighbors_dict = get_nodes_neighbors_dict(graph)
    assert (
        is_node_dominated(node=node_number, dominating_nodes=dominating_nodes, nodes_neighbors_dict=nodes_neighbors_dict) is True
    ), f"Node {node_number} should be linked to dominated set but is not."
    node_number = 4
    assert (
        is_node_dominated(node=node_number, dominating_nodes=dominating_nodes, nodes_neighbors_dict=nodes_neighbors_dict) is True
    ), f"Node {node_number} should be linked to dominated set but is not."
    node_number = 43
    assert (
        is_node_dominated(node=node_number, dominating_nodes=dominating_nodes, nodes_neighbors_dict=nodes_neighbors_dict) is False
    ), f"Node {node_number} shouldn't be linked to dominated set but is linked."
    dominating_nodes = {}
    node_number = 7
    assert (
        is_node_dominated(node=node_number, dominating_nodes=dominating_nodes, nodes_neighbors_dict=nodes_neighbors_dict) is False
    ), f"Node {node_number} shouldn't be linked to dominated set but is linked."


def test_construct_dominating_set() -> None:
    graph = load_graph(INPUT_DIR / GRAPH_FILENAME)
    graph.add_node(50, weight=1)
    graph.add_node(51, weight=5)
    graph.add_node(52, weight=30)
    graph.add_node(53, weight=40)
    graph.add_node(54)
    graph.add_edge(50, 9)
    graph.add_edge(51, 9)
    graph.add_edge(26, 52)
    graph.add_edge(26, 53)
    graph.add_edge(52, 53)

    nodes_weight_dict = dict(graph.nodes(data=True))
    nodes_neighbors_dict = get_nodes_neighbors_dict(graph=graph)
    nodes_to_remove, dominating_nodes, reduced_graph = reduce_graph(
        graph, nodes_weight_dict, nodes_neighbors_dict
    )
    dominating_set = construct_dominating_set(
        reduced_graph, dominating_nodes, nodes_weight_dict
    )
    assert nx.is_dominating_set(
        graph, dominating_set
    ), f"Set {dominating_set} is not a dominating set"


def test_dominant() -> None:
    graph = load_graph(INPUT_DIR / GRAPH_FILENAME)
    graph.add_node(50, weight=1)
    graph.add_node(51, weight=5)
    graph.add_node(52, weight=30)
    graph.add_node(53, weight=40)
    graph.add_node(54, weight=56)
    graph.add_edge(50, 9)
    graph.add_edge(51, 9)
    graph.add_edge(26, 52)
    graph.add_edge(26, 53)
    graph.add_edge(52, 53)

    assert nx.is_dominating_set(
        graph, dominant(graph)
    ), f"Set {dominant(graph)} is not a dominating set"
