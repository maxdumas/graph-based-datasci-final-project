import networkx as nx
from typing import Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def flow_layout(graph: nx.Graph, _s: Union[str, int] = "s", _t: Union[str, int] = "t") -> dict:
    """Position nodes other than the source and target in two straight lines.
    The left line includes all the successors of the source node and the right line
    includes all the predecessors of the target node. The source node is then placed
    to the left of the midpoint of the first line and the target node is placed to the
    right of the midpoint of the second line.

    Args:
        graph (nx.Graph): The flow network that must include the passed source node _s
        and the passed target node _t.
        _s (Union[str, int], optional): The source node. Defaults to "s".
        _t (Union[str, int], optional): The target node. Defaults to "t".

    Returns:
        dict: A dictionary of positions keyed by node. 

    Examples::
        >>> G = <a flow network>
        >>> pos = flow_layout(G, "s", "t")
        >>> nx.draw(G, pos=pos)

    """
    pos = nx.bipartite_layout(graph, graph.successors(_s))
    # t centers
    s_center = np.mean([pos[i][1] for i in graph.successors(_s)])
    t_center = np.mean([pos[i][1] for i in graph.predecessors(_t)])
    diff = t_center - s_center
    for i in graph.predecessors(_t):
        pos[i][1] -= diff
    t_center -= diff
    pos[_s] = [np.mean([pos[i][0]
                        for i in graph.successors(_s)]) - 1, s_center]
    pos[_t] = [np.mean([pos[i][0]
                        for i in graph.predecessors(_t)]) + 1, t_center]
    return pos


def draw_flow(graph: nx.DiGraph, flowDict: dict = None, fig_kwargs: dict = {}, nx_kwargs: dict = {}) -> None:
    """Draw the flow network highlighting the flows. 

    Args:
        graph (nx.DiGraph): The flow graph.
        flowDict (dict, optional): The flowDict. Defaults to None. If nothing is passed, 
        a flow dictionary will be calculated using nx.maximum_flow using source node
        "s", target node "t", and capacity attribute "capacity".
        fig_kwargs (dict, optional): Extra arguments to control the figure. Defaults to {}.
        nx_kwargs (dict, optional): Extra arguments to control the network figures. Defaults to {}.

    Examples::
        Without passing a flow dictionary:
            >>> draw_flow(G, fig_kwargs={"figsize":(10, 10)}, nx_kwargs={"font_size": 8})
        Without passing a flow dictionary:
            >>> G = <a flow network>
            >>> flowVal, flowDict = nx.maximum_flow(G, "s", "t", capacity="capacity")
            >>> draw_flow(G, flowDict=flowDict, fig_kwargs={"figsize":(10, 10)}, nx_kwargs={"font_size": 8})
    """
    if flowDict is None:
        flowVal, flowDict = nx.maximum_flow(
            graph, "s", "t", capacity="capacity")
    pos = nx.get_edge_attributes(graph, "pos")
    if not pos:
        pos = flow_layout(graph)
    H = nx.from_pandas_adjacency(
        pd.DataFrame(flowDict, index=graph.nodes()).fillna(
            0).T, create_using=nx.DiGraph
    )
    plt.figure(**fig_kwargs)
    nx.draw_networkx_nodes(graph, pos)
    nx.draw_networkx_labels(graph, pos)
    nx.draw_networkx_edges(graph, pos, edge_color="grey", alpha=0.15)
    nx.draw_networkx_edges(H, pos, edge_color="red")
    nx.draw_networkx_edge_labels(
        H, pos, edge_labels=nx.get_edge_attributes(H, "weight"), **nx_kwargs)
    plt.show()
