import networkx as nx


def get_nodes_weight_dict(
    graph: nx.Graph,
) -> {int: {str: int}}:
    return dict(graph.nodes(data=True))


def get_nodes_neighbors_dict(
    graph: nx.Graph,
) -> {int: set}:
    return {
        node: set([edge[1] for edge in list(graph.edges(node))]) for node in graph.nodes
    }


def get_dominated_nodes(
    graph: nx.Graph, dominating_nodes: {int}, nodes_neighbors_dict: {int: {int}}
) -> {int}:
    dominated_nodes = set()
    for node in graph.nodes:
        if is_node_dominated(
            node=node,
            dominating_nodes=dominating_nodes,
            nodes_neighbors_dict=nodes_neighbors_dict,
        ):
            dominated_nodes.add(node)
    return dominated_nodes


def get_non_dominated_nodes(
    graph: nx.Graph,
    dominating_nodes: {int},
    nodes_neighbors_dict: {int: {int}},
) -> {int}:
    non_dominated_nodes = set()
    for node in graph.nodes:
        if not is_node_dominated(
            node=node,
            dominating_nodes=dominating_nodes,
            nodes_neighbors_dict=nodes_neighbors_dict,
        ):
            non_dominated_nodes.add(node)
    return non_dominated_nodes


def is_node_dominated(
    node: int, dominating_nodes: {int}, nodes_neighbors_dict: {int: {int}}
) -> bool:
    is_linked = False
    if node in dominating_nodes:
        is_linked = True
    else:
        node_neighbors = nodes_neighbors_dict[node]
        for node_neighbor in node_neighbors:
            if node_neighbor in dominating_nodes:
                is_linked = True
    return is_linked


def get_max_frequency_score_node(
    graph: nx.Graph,
    dominating_nodes: {int},
    random_non_dominated_node: int,
    nodes_weight_dict: {int: {str: int}},
    dominated_nodes: {int},
    non_dominated_nodes: {int},
    nodes_neighbors_dict: {int: {int}},
) -> int:
    """
    Return the node in the neighborhood of random_non_dominated_node with the maximum frequency score
    """

    nodes_frequency_score_dict = get_nodes_frequency_score_dict(
        graph=graph,
        dominating_nodes=dominating_nodes,
        node=random_non_dominated_node,
        nodes_weight_dict=nodes_weight_dict,
        dominated_nodes=dominated_nodes,
        non_dominated_nodes=non_dominated_nodes,
        nodes_neighbors_dict=nodes_neighbors_dict,
    )
    max_frequency_score_node = max(
        nodes_frequency_score_dict.keys(), key=lambda k: nodes_frequency_score_dict[k]
    )
    return max_frequency_score_node


def get_nodes_frequency_score_dict(
    graph: nx.Graph,
    dominating_nodes: {int},
    node: int,
    nodes_weight_dict: {int: {str: int}},
    dominated_nodes: {int},
    non_dominated_nodes: {int},
    nodes_neighbors_dict: {int: {int}},
):
    nodes_frequency_score_dict = {
        node: get_frequency_score(
            graph=graph,
            dominating_nodes=dominating_nodes,
            node=node,
            dominated_nodes=dominated_nodes,
            non_dominated_nodes=non_dominated_nodes,
            nodes_weight_dict=nodes_weight_dict,
            nodes_neighbors_dict=nodes_neighbors_dict,
        )
        for node in nodes_neighbors_dict[node].union({node})
    }
    return nodes_frequency_score_dict


def get_frequency_score(
    graph: nx.Graph,
    dominating_nodes: {int},
    node: int,
    dominated_nodes: {int},
    non_dominated_nodes: {int},
    nodes_weight_dict: {int: {str: int}},
    nodes_neighbors_dict: {int: {int}},
) -> int:
    frequency_score = 0
    if node in dominating_nodes:
        # c1_nodes : dominated nodes that would become non dominated by removing node_number from dominating nodes

        dominating_nodes_reduced = dominating_nodes.copy()
        dominating_nodes_reduced.remove(node)

        c1_nodes = set()
        for dominated_node in dominated_nodes:
            if not is_node_dominated(
                node=dominated_node,
                dominating_nodes=dominating_nodes_reduced,
                nodes_neighbors_dict=nodes_neighbors_dict,
            ):
                c1_nodes.add(dominated_node)

        for c1_node in c1_nodes:
            frequency_score += (
                get_node_frequency(graph=graph, node=c1_node)
                / nodes_weight_dict[node]["weight"]
            )
    else:  # case where node_number is not a dominating node
        # c2_nodes : non dominated nodes that would become dominated by adding node_number into dominating nodes

        dominating_nodes_increased = dominating_nodes.copy()
        dominating_nodes_increased.add(node)

        c2_nodes = set()
        for non_dominated_node in non_dominated_nodes:
            if is_node_dominated(
                node=non_dominated_node,
                dominating_nodes=dominating_nodes_increased,
                nodes_neighbors_dict=nodes_neighbors_dict,
            ):
                c2_nodes.add(non_dominated_node)

        for c2_node in c2_nodes:
            frequency_score += (
                get_node_frequency(graph=graph, node=c2_node)
                / nodes_weight_dict[node]["weight"]
            )

    return frequency_score


def get_node_frequency(graph: nx.Graph, node: int) -> int:
    return 1


def compute_graph_dominating_set_score(
    dominating_set: {int}, nodes_weight_dict: {int: {str: int}}
) -> int:
    score = sum(
        [
            nodes_weight_dict[dominating_node]["weight"]
            for dominating_node in dominating_set
        ]
    )
    return score
