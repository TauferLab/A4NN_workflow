
import networkx as nx
import igraph as ig

def from_edgelist_to_nx(edgelist, create_using='directed', convert_labels_to_ints=False):
    """
    Convert an edgelist to a networkx object

    :param edgelist: list of tuples describing the network connections
    :param create_using: string to define the type of network. Only directed and undirected are supported 
    :param convert_labels_to_ints: bool if True convert node labels to integers

    :returns a networkx graph object 
    """

    if create_using not in ['directed', 'undirected']:
        raise NotImplementedError(f"Graph type {create_using} not supported")

    graph_type = nx.DiGraph if create_using == 'directed' else nx.Graph

    if isinstance(edgelist, list):
        G = nx.from_edgelist(edgelist, create_using=graph_type)
    else:
        raise TypeError(f"Edge list provided is not a list but a {type(edgelist)}")

    if convert_labels_to_ints:
        G = nx.convert_node_labels_to_integers(G, first_label=0, label_attribute='name')
    return G


def from_networkx_to_igraph(graph, node_attrs=None, edge_attrs=None):
    """
    Convert a networkX graph object to igraph
    :param graph: networkX object
    :param node_attrs: list of node attributes available
    :param edge_attrs: list of edge attributes available
    :return igraph object
    """

    # Make sure all nodes are integers. Otherwise, convert labels to integers
    flag_labels = all(isinstance(e, (int)) for e in graph.nodes)
    if not flag_labels: 
        graph = nx.convert_node_labels_to_integers(graph, first_label=0, label_attribute='name')
    
    if node_attrs:
        node_attrs.append('name')
    else:
        node_attrs = ['name']

    num_nodes = len(graph)
    # Convert networkx to igraph
    g = ig.Graph(num_nodes, list(zip(*list(zip(*nx.to_edgelist(graph)))[:2])), directed=graph.is_directed())
    
    # Add node attributes to igraph nodes if any
    for attr in node_attrs:
        values = list(nx.get_node_attributes(graph, name=attr).values())
        if len(values) > 0:
            g.vs[:][attr] = values

    # Add edge attributes to igraph nodes if any
    if edge_attrs:
        for attr in edge_attrs:
            values = list(nx.get_edge_attributes(graph, name=attr).values())
            if len(values) > 0:
                g.es[:][attr] = values
    
    return g
