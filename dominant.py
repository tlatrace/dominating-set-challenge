import sys
import os
import time
import networkx as nx
import operator


def dominant(graph) -> list:
    """
    A Faire:
    - Ecrire une fonction qui retourne le dominant du graphe non dirigé g passé en parametre.
    - cette fonction doit retourner la liste des noeuds d'un petit dominant de g

    :param graph: le graphe est donné dans le format networkx : https://networkx.github.io/documentation/stable/reference/classes/graph.html

    """
    nodes_dict = {
        node_number: {
            "weight": weight_dict["weight"],
            "neighbors": get_node_neighbors(graph=graph, node_number=node_number),
            "ratio": len(get_node_neighbors(graph=graph, node_number=node_number)) / weight_dict["weight"]
        }
        for node_number, weight_dict in dict(graph.nodes(data=True)).items()
    }

    nodes_ratio_dict = {
        node_number: node_dict["ratio"]
        for node_number, node_dict in nodes_dict.items()
    }

    sorted_nodes_ratio_list = sorted(nodes_ratio_dict.items(), key=operator.itemgetter(1), reverse=True)
    sorted_nodes_list = [node_tuple[0] for node_tuple in sorted_nodes_ratio_list]

    dominating_set = list()
    counter = 0
    while (not nx.is_dominating_set(graph, dominating_set)) and (counter <= len(list(graph.nodes))):
        next_node = sorted_nodes_list[counter]
        if not is_node_linked_to_dominating_set(next_node, graph, dominating_set):
            dominating_set.append(next_node)
        counter += 1

    return sorted(dominating_set)


def get_node_neighbors(
    graph: nx.Graph,
    node_number: int,
) -> list:
    return [edge[1] for edge in list(graph.edges(node_number))]


def is_node_linked_to_dominating_set(
        node: int,
        graph: nx.Graph,
        dominating_set: list,
) -> bool:
    for dominating_node in dominating_set:
        graph_node_neighbors = get_node_neighbors(graph, dominating_node)
        if node in graph_node_neighbors:
            return True
    return False


#########################################
#### Ne pas modifier le code suivant ####
#########################################


def load_graph(name):
    with open(name, "r") as f:
        state = 0
        G = None
        for l in f:
            if state == 0:  # Header nb of nodes
                state = 1
            elif state == 1:  # Nb of nodes
                nodes = int(l)
                state = 2
            elif state == 2:  # Header position
                i = 0
                state = 3
            elif state == 3:  # Position
                i += 1
                if i >= nodes:
                    state = 4
            elif state == 4:  # Header node weight
                i = 0
                state = 5
                G = nx.Graph()
            elif state == 5:  # Node weight
                G.add_node(i, weight=int(l))
                i += 1
                if i >= nodes:
                    state = 6
            elif state == 6:  # Header edge
                i = 0
                state = 7
            elif state == 7:
                if i > nodes:
                    pass
                else:
                    edges = l.strip().split(" ")
                    for j, w in enumerate(edges):
                        w = int(w)
                        if w == 1 and (not i == j):
                            G.add_edge(i, j)
                    i += 1

        return G


#########################################
#### Ne pas modifier le code suivant ####
#########################################
if __name__ == "__main__":
    input_dir = os.path.abspath(sys.argv[1])
    output_dir = os.path.abspath(sys.argv[2])

    # un repertoire des graphes en entree doit être passé en parametre 1
    if not os.path.isdir(input_dir):
        print(input_dir, "doesn't exist")
        exit()

    # un repertoire pour enregistrer les dominants doit être passé en parametre 2
    if not os.path.isdir(output_dir):
        print(input_dir, "doesn't exist")
        exit()

    # fichier des reponses depose dans le output_dir et annote par date/heure
    output_filename = "answers_{}.txt".format(
        time.strftime("%d%b%Y_%H%M%S", time.localtime())
    )
    output_file = open(os.path.join(output_dir, output_filename), "w")

    for graph_filename in sorted(os.listdir(input_dir)):
        # importer le graphe
        g = load_graph(os.path.join(input_dir, graph_filename))

        # calcul du dominant
        D = sorted(dominant(g), key=lambda x: int(x))

        # ajout au rapport
        output_file.write(graph_filename)
        for node in D:
            output_file.write(" {}".format(node))
        output_file.write("\n")

    output_file.close()
