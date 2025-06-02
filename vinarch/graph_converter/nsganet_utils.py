
import pydot
from pydot import Node, Edge

import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import rgb2hex

# The function for building the dependency graph is adapted from: https://github.com/ianwhale/nsga-net/
def build_dependency_graph(gene):
    """
    Build a graph describing the connections of a phase.
    "Repairs" made are as follows:
        - If a node has no input, but gives output, connect it to the input node (index 0 in outputs).
        - If a node has input, but no output, connect it to the output node (value returned from forward method).
    :param gene: gene describing the phase connections represented as a nested list of binary values.
    :return: dict describing the edge list for a given phase
    """
    graph = {} # dictionary to store the edge list
    residual = gene[-1][0] == 1 # :boolean True if residual connection is 1, false otherwise
    
    # First pass, build the graph without repairs.
    graph[1] = [] # define a list of connections for node 1
    for i in range(len(gene) - 1): # iterate over the computational units, skip the residual bit 
        graph[i + 2] = [j + 1 for j in range(len(gene[i])) if gene[i][j] == 1] # skip node 1, and iterate list of neighbors
    
    graph[len(gene) + 1] = [0] if residual else [] # When processing the output, add a connection to the input if residual is True, otherwise skip
    
    # Determine which nodes, if any, have no inputs and/or outputs.
    no_inputs = []
    no_outputs = []
    for i in range(1, len(gene) + 1):
        if len(graph[i]) == 0: # if the edge list for current node is empty, then the node has no inputs
            no_inputs.append(i)
    
        has_output = False
        for j in range(i + 1, len(gene) + 2): # if the current node is in the edge list of subsequent nodes, then it has at least one output to another computational unit
            if i in graph[j]:
                has_output = True
                break
    
        if not has_output: # Otherwise, the current node has no outputs
            no_outputs.append(i)
    
    for node in no_outputs:
        if node not in no_inputs:
            # No outputs, but has inputs. Connect to output node.
            graph[len(gene) + 1].append(node)
    
    for node in no_inputs:
        if node not in no_outputs:
            # No inputs, but has outputs. Connect to input node.
            graph[node].append(0)
    
    return graph


def build_genome_structure(genome):
    """
    Returns a a list of dict objects containing relevant information for each NN phase
    :param genome: list of lists
    :returns list of dict objects
    """
    structure = [] # store the complete graph structure
    phase_count = 0
    for i, gene in enumerate(genome): # iterate over each phase
        all_zeros = sum([sum(t) for t in gene[:-1]]) == 0 # check if the entire phase has no computational units
        if all_zeros:
            continue # skip the phase completely since the gene is all zeros

        prefix = f"phase_{str(phase_count)}" # ID of phase
        pool = prefix + "_pool" # ID of pooling layer

        # Create a unique id for each node in the current phase
        nodes = [prefix + "_node_0"] + [prefix+f"_node_{str(node+1)}" for node in range(len(gene) + 1)]
        
        graph = build_dependency_graph(gene) # extract the corresponding edgelist dictionary
        edges = [] # store list of edges
        
        for source, neighbors in graph.items(): # iterate over the edgelists. The connection must be neighbor -> source
            for neighbor in neighbors:
                edges.append((nodes[neighbor], nodes[source]))

        structure.append({
            "nodes": nodes,
            "edges": edges,
            "pool": pool,
            "all_zeros": all_zeros,
            "phase":phase_count,
            "graph":graph
        })

        phase_count += 1 # increment the number of phases
    return structure 


# This function is adapted from the make_dot_genome() function: https://github.com/ianwhale/nsga-net/blob/master/visualization/macro_visualize.py
def build_genome_edgelist(genome):
    """
    Returns an edgelist for the complete computational graph of a given genome/NN architecture
    :param genome: list of lists
    :return list of tuples
    """
    # Create the structure for each phase
    structure = build_genome_structure(genome)

    # Create an edgelist
    edgelist = []

    # Iterate over the list of structures
    for struct in structure:
        nodes = struct["nodes"]
        edges = struct["edges"]
        phase = struct["phase"]
        pool = struct["pool"]
        graph = struct["graph"]
        all_zeros = struct["all_zeros"]

        if phase > 0: # For phases other than the first one, make a connection between previous pooling layer and the input node
            edgelist.append((structure[phase - 1]['pool'], nodes[0]))

        # Add edges
        edgelist.extend(edges)
            
        # add connection from output node to pooling layer
        edgelist.append((nodes[-1], pool))
    # Connect the last pooling layer to a linear (FC) layer
    edgelist.append((structure[-1]['pool'], 'linear'))

    return edgelist


# This function is adapted from: https://github.com/ianwhale/nsga-net/blob/master/visualization/macro_visualize.py
def build_pydot_genome(genome, **kwargs):
    """
    Returns a pyDot object of a given genome

    :param genome: a nested list representation of the NN architecture
    :param **kwargs: additional properties for graph visualization as a dict or keyword arguments
    """

    # default dictionary of node attributes
    node_attr = dict(style='filled',
                     align='left',
                     fontsize='20',
                     ranksep='0.1',
                     height='0.1')
    
    if not isinstance(genome, list): # check correct input
        raise TypeError(f"Edge list provided is not a list but a {type(genome)}")

    # Extract genome structure
    structure = build_genome_structure(genome)

    num_phases = len(structure) #number of phases
    cm = colormaps.get_cmap(kwargs['node_cmap'])
    node_colors = [rgb2hex(cm(i)) for i in range(num_phases)] # generate a different color for each phase

    # Create the pyDot graph
    dot = pydot.Dot(graph_type='digraph', rankdir=kwargs['rankdir'])
    if kwargs['title']: # set a title if any 
        dot.set_graph_defaults(label=kwargs['title']+"\n")
        dot.set_graph_defaults(labelloc='t')

    dot.add_node(Node("input", label="Input", color='salmon', shape=kwargs['linear_shape'], **node_attr)) # create input node

    # Create nodes and edges for each phase
    for j, struct in enumerate(structure):
        nodes = struct['nodes']
        edges = struct['edges']
        phase = struct['phase']
        pool = struct['pool']
        graph = struct['graph']
        all_zeros = struct['all_zeros']
    
        # add nodes
        dot.add_node(Node(nodes[0], label=nodes[0].split("_")[-1], shape=kwargs["io_shape"],
                     fillcolor=kwargs["input_color"], **node_attr))

        if phase > 0:
            dot.add_edge(Edge(structure[phase-1]['pool'], nodes[0]))

        # Add all nodes within a phase to the same cluster
        if not all_zeros:
            cluster = pydot.Cluster(f"cluster_{phase}", 
                                    label='', 
                                    fillcolor=kwargs['phase_background_color'],
                                    fontcolor='black',
                                    style='filled'
                                   )

            for i in range(1, len(nodes) - 1): # ignore the input and output CNNs
                if len(graph[i]) != 0: # if there exist connections to parents
                    cluster.add_node(Node(nodes[i], 
                                          label=nodes[i].split("_")[-1],
                                          fillcolor=node_colors[phase], 
                                          shape=kwargs["node_shape"],
                                          **node_attr
                                         ))
            dot.add_subgraph(cluster)

        # Add the output node
        dot.add_node(Node(nodes[-1], label=nodes[-1].split("_")[-1], shape=kwargs["io_shape"], 
                          fillcolor=kwargs['output_color'], **node_attr))
        # Add pooling node
        dot.add_node(Node(pool, label='Pooling', 
                          fillcolor=kwargs['pool_color'], shape=kwargs["pool_shape"], **node_attr))
        
        # Add edges
        for e1, e2 in edges:
            dot.add_edge(Edge(e1, e2))

        dot.add_edge(Edge(nodes[-1], pool)) # add edge between the output node and the pooling layer

    dot.add_edge(Edge("input", structure[0]['nodes'][0])) # add edge between input and the first phase
    dot.add_node(Node("linear", label="Linear", fillcolor=kwargs['fc_color'], shape=kwargs["linear_shape"], **node_attr))
    dot.add_edge(Edge(structure[-1]['pool'], "linear")) # add edge between last pooling layer and linear layer    
    
    return dot
