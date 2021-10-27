import networkx as nx


def weighted_degree_0_rule(
    graph: nx.Graph, nodes_neighbors_dict: {int: {int}}
) -> {int}:
    dominating_nodes = set()
    for node in graph.nodes:
        if len(nodes_neighbors_dict[node]) == 0:  # node of degree 0
            dominating_nodes.add(node)
    return dominating_nodes


def weighted_degree_1_rule_1(
    graph: nx.Graph,
    nodes_weight_dict: {int: {str: int}},
    nodes_neighbors_dict: {int: {int}},
) -> ({int}, {int}):
    dominating_nodes = set()
    nodes_to_remove = set()
    for node in graph.nodes:
        node_neighbors = nodes_neighbors_dict[node]
        node_neighbors_of_degree_1 = [
            node_neighbor
            for node_neighbor in node_neighbors
            if len(nodes_neighbors_dict[node_neighbor]) == 1  # node of degree 1
        ]
        if len(node_neighbors_of_degree_1) == 1:  # check if has a neighbor
            node_neighbor = node_neighbors_of_degree_1[0]
            if (
                nodes_weight_dict[node_neighbor]["weight"]
                > nodes_weight_dict[node]["weight"]
            ):
                dominating_nodes.add(node)
                for node_to_remove in node_neighbors_of_degree_1:
                    nodes_to_remove.add(node_to_remove)
    return dominating_nodes, nodes_to_remove


def weighted_degree_1_rule_2(
    graph: nx.Graph,
    nodes_weight_dict: {int: {str: int}},
    nodes_neighbors_dict: {int: {int}},
) -> ({int}, {int}):
    dominating_nodes = set()
    nodes_to_remove = set()
    for node in graph.nodes:
        node_neighbors = nodes_neighbors_dict[node]
        node_neighbors_of_degree_1 = [
            node_neighbor
            for node_neighbor in node_neighbors
            if len(nodes_neighbors_dict[node_neighbor]) == 1  # node of degree 1
        ]
        if node_neighbors_of_degree_1:  # check if some neighbor have degree 1
            neighbors_weight_sum = get_neighbors_weight_sum(
                nodes_list=node_neighbors_of_degree_1,
                nodes_weight_dict=nodes_weight_dict,
            )
            if neighbors_weight_sum > nodes_weight_dict[node]["weight"]:
                dominating_nodes.add(node)
                for node_to_remove in node_neighbors_of_degree_1:
                    nodes_to_remove.add(node_to_remove)
    return dominating_nodes, nodes_to_remove


def get_neighbors_weight_sum(
    nodes_list: [int], nodes_weight_dict: {int: {str: int}}
) -> int:
    neighbors_weight_sum = 0
    for node_neighbor in nodes_list:
        neighbors_weight_sum += nodes_weight_dict[node_neighbor]["weight"]
    return neighbors_weight_sum


def weighted_degree_2_rule(
    graph: nx.Graph,
    nodes_weight_dict: {int: {str: int}},
    nodes_neighbors_dict: {int: {int}},
) -> ({int}, {int}):
    dominating_nodes = set()
    nodes_to_remove = set()
    for node in graph.nodes:
        if len(nodes_neighbors_dict[node]) == 2:  # node of degree 2
            node_neighbors = nodes_neighbors_dict[node]
            for node_neighbor in node_neighbors:
                if len(nodes_neighbors_dict[node_neighbor]) == 2:
                    second_neighbor = next(iter(node_neighbors - {node_neighbor}))
                    if second_neighbor in nodes_neighbors_dict[node_neighbor]:
                        if (
                            nodes_weight_dict[node_neighbor]["weight"]
                            > nodes_weight_dict[second_neighbor]["weight"]
                            and nodes_weight_dict[node]["weight"]
                            > nodes_weight_dict[second_neighbor]["weight"]
                        ):
                            dominating_nodes.add(second_neighbor)
                            nodes_to_remove.add(node_neighbor)
                            nodes_to_remove.add(node)
    return dominating_nodes, nodes_to_remove
