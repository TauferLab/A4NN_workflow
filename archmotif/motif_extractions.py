import re
from collections import defaultdict
from itertools import combinations

import networkx as nx
import pandas as pd


class EvoGraphAnalyzer:

    def __init__(self, graph: nx.DiGraph):
        self.graph = graph
        self.phases = self._extract_phases()

    # ------------------------------------------------------------------
    # Phase extraction
    # ------------------------------------------------------------------

    def _extract_phases(self):
        """
        Extract phase subgraphs from a graph whose nodes follow:

            phase_0_node_0
            phase_0_node_1
            phase_1_node_0
            ...

        Returns:
            {
                0: phase0_subgraph,
                1: phase1_subgraph,
                ...
            }
        """

        phase_nodes = defaultdict(list)

        for node in self.graph.nodes:
            match = re.match(
                r"phase_(\d+)_node_(\d+)",
                str(node)
            )

            if match:
                phase_id = int(match.group(1))
                phase_nodes[phase_id].append(node)

        return {
            phase_id: self.graph.subgraph(nodes).copy()
            for phase_id, nodes in phase_nodes.items()
        }

    @staticmethod
    def local_id(node):
        return int(str(node).split("_")[-1])

    # ------------------------------------------------------------------
    # Motif detectors
    # ------------------------------------------------------------------

    @staticmethod
    def has_triangle(G):
        for a in G.nodes:
            for b in G.successors(a):
                for c in G.successors(b):
                    if G.has_edge(a, c):
                        return True

        return False

    @staticmethod
    def has_diamond(G):
        def node_index(name):
            match = re.search(r'(\d+)$', name)
            return int(match.group(1)) if match else None

        indices = [
            node_index(n)
            for n in G.nodes
            if 'pool' not in n and 'linear' not in n
        ]
        indices = [i for i in indices if i is not None]
        max_index = max(indices) if indices else None

        excluded = set()
        for n in G.nodes:
            if 'pool' in n or 'linear' in n:
                excluded.add(n)
                continue
            idx = node_index(n)
            if idx is not None and idx == max_index:
                excluded.add(n)

        valid_nodes = [n for n in G.nodes if n not in excluded]

        for a in valid_nodes:
            children = [c for c in G.successors(a) if c not in excluded]

            for b, c in combinations(children, 2):
                b_children = set(x for x in G.successors(b) if x not in excluded)
                c_children = set(x for x in G.successors(c) if x not in excluded)

                if b_children & c_children:
                    return True

        return False

    # @staticmethod
    # def has_diamond(G):

        for a in G.nodes:

            children = list(G.successors(a))

            for b, c in combinations(children, 2):

                b_children = set(G.successors(b))
                c_children = set(G.successors(c))

                if b_children & c_children:
                    return True

        return False

    @staticmethod
    def is_chain(G):

        active_nodes = [
            n
            for n in G.nodes
            if G.in_degree(n) > 0 or G.out_degree(n) > 0
        ]

        if len(active_nodes) <= 1:
            return True

        H = G.subgraph(active_nodes)

        return (
            len(nx.dag_longest_path(H))
            == len(active_nodes)
        )

    @classmethod
    def fork_no_join_before_output(cls, G):

        indices = [
            cls.local_id(n) for n in G.nodes
            if cls.local_id(n) is not None
        ]

        if not indices:
            return False

        max_index = max(indices)

        outputs = [
            n for n in G.nodes
            if cls.local_id(n) == max_index
        ]

        if not outputs:
            return False

        for node in G.nodes:

            idx = cls.local_id(node)

            if idx is not None and idx < max_index:

                if G.in_degree(node) > 1:
                    return False

        return True
    # @classmethod
    # def fork_no_join_before_output(cls, G):

    #     outputs = [
    #         n for n in G.nodes
    #         if cls.local_id(n) == 5
    #     ]

    #     if not outputs:
    #         return False

    #     output = outputs[0]

    #     for node in G.nodes:

    #         if G.out_degree(node) > 1:

    #             reach_sets = [
    #                 nx.descendants(G, child)
    #                 for child in G.successors(node)
    #             ]

    #             for a, b in combinations(reach_sets, 2):

    #                 joins = (a & b) - {output}

    #                 if joins:
    #                     return False

    #             return True

    #     return False

    # ------------------------------------------------------------------
    # Features for one phase
    # ------------------------------------------------------------------

    @classmethod
    def phase_features(cls, phase_graph):

        local_ids = {
            cls.local_id(n)
            for n in phase_graph.nodes
        }

        node0 = next(
            (n for n in phase_graph.nodes
             if cls.local_id(n) == 0),
            None
        )

        node5 = next(
            (n for n in phase_graph.nodes
             if cls.local_id(n) == 5),
            None
        )

        return {

            "triangle":
                cls.has_triangle(phase_graph),

            "diamond":
                cls.has_diamond(phase_graph),

            "line":
                cls.is_chain(phase_graph),

            "only_merge_at_output":
                cls.fork_no_join_before_output(
                    phase_graph
                ),

            "input_to_output_edge":
                (
                    phase_graph.has_edge(node0, node5)
                    if node0 and node5
                    else False
                ),

            "output_merge_count":
                (
                    phase_graph.in_degree(node5)
                    if node5
                    else 0
                ),

            "input_fork_count":
                (
                    phase_graph.out_degree(node0)
                    if node0
                    else 0
                ),

            "missing_nodes":
                sorted(
                    set(range(6)) - local_ids
                ),
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self):
        """
        Returns:

        {
            phase_id: feature_dict
        }
        """

        return {
            phase_id: self.phase_features(subgraph)
            for phase_id, subgraph in sorted(
                self.phases.items()
            )
        }

    def phase_summary(self):
        """
        Returns one record per phase.
        """

        results = []

        for phase_id, subgraph in sorted(
            self.phases.items()
        ):

            results.append({
                "phase": phase_id,
                "subgraph": subgraph,
                "features": self.phase_features(subgraph)
            })

        return results

    def feature_table(self):
        """
        Returns a DataFrame-friendly list.

        Best for correlation and ML.
        """

        rows = []

        for phase_id, subgraph in sorted(
            self.phases.items()
        ):

            features = self.phase_features(subgraph)

            rows.append({
                "phase": phase_id,
                **features
            })

        return rows

    def feature_dataframe(self):
        return pd.DataFrame(
            self.feature_table()
        )
    
def find_extreme_graphs(nx_graphs):
    """
    Find one graph index for each extreme metric.

    Assumes node names look like:
        phase_0_node_0
        phase_0_node_1
        phase_1_pool
        ...

    Parameters
    ----------
    nx_graphs : list[nx.Graph]

    Returns
    -------
    dict
    """

    max_phases = -1
    max_phases_idx = None

    max_phase_nodes = -1
    max_phase_nodes_idx = None
    max_phase_nodes_phase = None

    max_phase_edges = -1
    max_phase_edges_idx = None
    max_phase_edges_phase = None

    max_node0_degree = -1
    max_node0_degree_idx = None
    max_node0_degree_phase = None

    for idx, G in enumerate(nx_graphs):

        # discover phases
        phases = sorted({
            int(m.group(1))
            for n in G.nodes()
            if (m := re.match(r"phase_(\d+)_", str(n)))
        })

        # -------------------------
        # Metric 1: max # phases
        # -------------------------
        n_phases = len(phases)

        if n_phases > max_phases:
            max_phases = n_phases
            max_phases_idx = idx

        for phase in phases:

            phase_prefix = f"phase_{phase}_"

            phase_nodes = [
                n for n in G.nodes()
                if str(n).startswith(phase_prefix)
            ]

            n_nodes = len(phase_nodes)

            phase_subgraph = G.subgraph(phase_nodes)
            n_edges = phase_subgraph.number_of_edges()

            # -------------------------
            # Metric 2a: max nodes
            # -------------------------
            if n_nodes > max_phase_nodes:
                max_phase_nodes = n_nodes
                max_phase_nodes_idx = idx
                max_phase_nodes_phase = phase

            # -------------------------
            # Metric 2b: max edges
            # -------------------------
            if n_edges > max_phase_edges:
                max_phase_edges = n_edges
                max_phase_edges_idx = idx
                max_phase_edges_phase = phase

            # -------------------------
            # Metric 3: max degree of node_0
            # -------------------------
            node0 = f"phase_{phase}_node_0"

            if node0 in G:
                deg = G.degree(node0)

                if deg > max_node0_degree:
                    max_node0_degree = deg
                    max_node0_degree_idx = idx
                    max_node0_degree_phase = phase

    return {
        "max_phases": {
            "value": max_phases,
            "graph_idx": max_phases_idx,
        },
        "max_nodes_in_phase": {
            "value": max_phase_nodes,
            "graph_idx": max_phase_nodes_idx,
            "phase": max_phase_nodes_phase,
        },
        "max_edges_in_phase": {
            "value": max_phase_edges,
            "graph_idx": max_phase_edges_idx,
            "phase": max_phase_edges_phase,
        },
        "max_node0_degree": {
            "value": max_node0_degree,
            "graph_idx": max_node0_degree_idx,
            "phase": max_node0_degree_phase,
        },
    }