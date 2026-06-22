import networkx as nx
import torch
from random import randint
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Arc
import numpy as np
import os

from torch_geometric.data import Data
from torch_geometric.utils import remove_isolated_nodes, to_networkx, subgraph

try:
    from torch_geometric.utils import dropout_edge
except ImportError:
    from torch_geometric.utils import dropout_adj as dropout_edge

from torch_scatter import scatter_sum, scatter_add

from GOOD.utils.splitting import split_graph
from GOOD.definitions import ROOT_DIR

edge_colors = {
    "inv": "green",
    "spu": "black",
    # "inv": "black",
    # "spu": "green",
    "added": "red"
}
node_colors = {
    True: "red",
    False: "#1f78b4"
}


def remove_from_graph(G, edge_index_to_remove=None, what_to_remove=None):
    if edge_index_to_remove is None and what_to_remove:
        G = G.copy()
        edge_remove = []
        for (u,v), val in nx.get_edge_attributes(G, 'origin').items():
            if val == what_to_remove:
                edge_remove.append((u,v))
        G.remove_edges_from(edge_remove)
        G.remove_edges_from([(v,u) for v,u in G.edges() if not G.has_edge(u,v)])
        G.remove_nodes_from(list(nx.isolates(G)))
    elif not edge_index_to_remove is None:
        G = G.copy()
        G.remove_edges_from([(u.item(), v.item()) for u,v in edge_index_to_remove.T])
        G.remove_edges_from([(v,u) for v,u in G.edges() if not G.has_edge(u,v)])
        G.remove_nodes_from(list(nx.isolates(G)))
    else:
        raise ValueError(f"what_to_remove can not be None with edge_index None")
    return G

def mark_edges(G, inv_edge_index, spu_edge_index, inv_edge_w=None, spu_edge_w=None):
    nx.set_edge_attributes(
        G,
        name="origin",
        values={(u.item(), v.item()): "inv" for u,v in inv_edge_index.T}
    )
    if not inv_edge_w is None:
        d = {(u.item(), v.item()): round(inv_edge_w[i].item(),2) for i, (u,v) in enumerate(inv_edge_index.T)}
        # assert np.all([d[u,v] == d[v,u] for u,v in d.keys()])
        nx.set_edge_attributes(
            G,
            name="attn_weight",
            values=d
        )
    nx.set_edge_attributes(
        G,
        name="origin",
        values={(u.item(), v.item()): "spu" for u,v in spu_edge_index.T}
    )
    if not spu_edge_w is None:
        d = {(u.item(), v.item()): round(spu_edge_w[i].item(),2) for i, (u,v) in enumerate(spu_edge_index.T)}
        assert np.all([d[u,v] == d[v,u] for u,v in d.keys()])
        nx.set_edge_attributes(
            G,
            name="attn_weight",
            values=d
        )

def mark_frontier(G, G_filt):
    # mark frontier nodes as nodes attached to both inv and spu parts
    # to mark nodes check which nodes have a change in the degree between original and filtered graph
    # frontier = []
    # for n in G_filt.nodes():
    #     if G.degree[n] != G_filt.degree[n]:                    
    #         frontier.append(n)            
    
    frontier = list(filter(lambda n: G.degree[n] != G_filt.degree[n], G_filt.nodes()))

    nx.set_node_attributes(G_filt, name="frontier", values=False)
    nx.set_node_attributes(G_filt, name="frontier", values={n: True for n in frontier})
    return len(frontier)

# def draw(config, G, name, subfolder="", pos=None, save=True, figsize=(6.4, 4.8), nodesize=350, with_labels=True, title=None, ax=None):
#     plt.figure(figsize=figsize)

#     if pos is None:
#         pos = nx.kamada_kawai_layout(G)

#     edge_color = list(map(lambda x: edge_colors[x], nx.get_edge_attributes(G,'origin').values()))
#     node_gt = list(nx.get_node_attributes(G, "node_gt").values())
#     # edge_color = list(nx.get_edge_attributes(G, "attn_weight").values())
#     # edge_color = ["red" if e > 0.90 else "black" for e in edge_color]
#     # nx.draw_networkx_edges(
#     #     G,
#     #     pos=pos,
#     #     edge_color="black"
#     # )
#     nx.draw(
#         G,
#         with_labels=with_labels,
#         pos=pos,
#         ax=ax,
#         node_size=nodesize,
#         node_color = ['lightgreen' if node_gt[i] else 'orange' for i in range(len(node_gt))],
#         # node_color=list(map(lambda x: node_colors[x], [nx.get_node_attributes(G,'frontier').get(n, False) for n in G.nodes()])),
#         # edgelist=[e for i, e in enumerate(G.edges()) if edge_color[i] > 0.5],
#         edge_color=edge_color,
#         # edge_cmap=plt.cm.Reds,
#     )

#     # Annotate with edge scores
#     if nx.get_edge_attributes(G, 'attn_weight') != {}:
#         nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'attn_weight'), font_size=6, alpha=0.8)
    
#     title = title if title is not None else f"Selected {sum([e == 'green' for e in edge_color])} relevant edges"
#     plt.title(title)
#     # print(f"Selected {sum([e == 'green' for e in edge_color])} relevant edges over {len(G.edges())}")

#     if save:
#         path = f'GOOD/kernel/pipelines/plots/{subfolder}/{config.load_split}_{config.util_model_dirname}/'
#         if not os.path.exists(path):
#             try:
#                 os.makedirs(path)
#             except Exception as e:
#                 print(e)
#                 exit(e)
#         plt.savefig(f'{path}/{name}.png')
#     else:
#         plt.show()

#     if ax is None:
#         plt.close()
#     return pos

# def draw_antonio(config, d, pos, subfolder, name, save=True):
#     what_take = "mask"
#     g1 = to_networkx(d, edge_attrs=["causal_mask", "mask"], to_undirected=False)
#     e_col = []
#     for e in g1.edges():
#         if g1.edges()[e][what_take]:
#             e_col.append("green")
#         else:
#             e_col.append("black")
    
#     if pos is None:
#         pos = nx.kamada_kawai_layout(g1)

#     plt.figure(figsize=(10,10))
    
#     # Draw nodes
#     nx.draw_networkx_nodes(g1, pos, node_color='lightblue', node_size=40)
#     nx.draw_networkx_labels(g1, pos, )

#     # Draw labels
#     # Draw directed edges with rounded style to show bidirectionality
#     nx.draw_networkx_edges(
#         g1, 
#         pos,
#         edgelist=g1.edges(),
#         edge_color= e_col,
#         arrowstyle='->',
#         arrowsize=20,
#         connectionstyle='arc3,rad=0.1'  # The 'rad' parameter makes edges curved
#     )

#     # Display the graph
#     if save:
#         path = f'GOOD/kernel/pipelines/plots/{subfolder}/{config.load_split}_{config.util_model_dirname}/'
#         if not os.path.exists(path):
#             try:
#                 os.makedirs(path)
#             except Exception as e:
#                 print(e)
#                 exit(e)
#         plt.savefig(f'{path}/{name}.png')
#     else:
#         plt.show()
#     return pos

# def draw_topk(config, G, name, k, subfolder="", pos=None):
#     if pos is None:
#         pos = nx.kamada_kawai_layout(G)

#     w = sorted(list(nx.get_edge_attributes(G, 'attn_weight').values()), reverse=True)
#     edge_color = []
#     for e in G.edges():
#         if G.edges[e]["attn_weight"] >= w[k]:
#             edge_color.append("green")
#         else:
#             edge_color.append("blue")

#     nx.draw(
#         G,
#         with_labels = True,
#         pos=pos,
#         edge_color=edge_color
#     )
#     # nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'attn_weight'), font_size=6, alpha=0.8)
#     path = f'GOOD/kernel/pipelines/plots/{subfolder}/{config.load_split}_{config.util_model_dirname}/'
#     if not os.path.exists(path):
#         try:
#             os.makedirs(path)
#         except Exception as e:
#             exit(e)
#     plt.savefig(f'{path}/{name}.png')
#     plt.close()
#     return pos

# def draw_gt(config, G, name, gt, edge_index, subfolder="", pos=None):
#     if pos is None:
#         pos = nx.kamada_kawai_layout(G)

#     edge_color = {}
#     for i in range(len(gt)):
#         (u,v) = edge_index.T[i]
#         if gt[i]:            
#             edge_color[(u.item(), v.item())] = "green"
#         else:
#             edge_color[(u.item(), v.item())] = "blue"
#     nx.draw(
#         G,
#         with_labels = True,
#         pos=pos,
#         edge_color=[edge_color[(u,v)] for u,v in G.edges()]
#     )
#     path = f'GOOD/kernel/pipelines/plots/{subfolder}/{config.load_split}_{config.util_model_dirname}/'
#     if not os.path.exists(path):
#         try:
#             os.makedirs(path)
#         except Exception as e:
#             exit(e)
#     plt.savefig(f'{path}/{name}.png')
#     plt.close()
#     return pos

def get_color_based_on_dataset(config, x):
    if "BAColorGV" in config.dataset.dataset_name or "BAColorRB" in config.dataset.dataset_name:
        # elif node_attr[i] == [1., 0., 0., 0.]:
        #     node_colors.append("red")
        # elif node_attr[i] == [0., 1., 0., 0.]:
        #     node_colors.append("blue")
        # elif node_attr[i] == [0., 0., 1., 0.]:
        #     node_colors.append("green")
        # elif node_attr[i] == [0., 0., 0., 1.]:
        #     node_colors.append("violet")
        if np.argmax(x) == 0:
            return "red"
        elif np.argmax(x) == 1:
            return "blue"
        elif np.argmax(x) == 2:
            return "green"
        elif np.argmax(x) == 3:
            return "violet"
        else:
            return "orange"
    elif config.dataset.dataset_name in ["MNIST", "CPatchMNIST", "CPatchMNIST2"]:
        return x[:3]
    elif config.dataset.dataset_name == "MUTAG":
        atom_type = np.argmax(x)  # Convert one-hot to indices
        # color_map = {atype: plt.cm.tab10(i % 10) for i, atype in enumerate(unique_atom_types)}
        # node_colors = [color_map[atype] for atype in atom_types]
        return plt.cm.tab10(atom_type % 14)
    elif config.dataset.dataset_name == "GraphSST2Planted":
        return "lightblue"

def draw_colored(config, G, name, thrs=None, node_expl=None, edge_expl="", subfolder="", pos=None, save=True, figsize=(6.4, 4.8), nodesize=150, with_labels=True, title=None, ax=None, topk=None):
    plt.figure(figsize=figsize)

    node_gt = list(nx.get_node_attributes(G, "node_gt").values())
    node_attr = list(nx.get_node_attributes(G, "x").values())
    
    if pos is None and config.dataset.dataset_name not in ["MNIST", "CPatchMNIST", "CPatchMNIST2", "GraphSST2Planted"]:
        pos = nx.kamada_kawai_layout(G)
        # pos[len(G.nodes)-2] = np.array([0.8, 0.8]) # useful for BAColorGV
        # pos[len(G.nodes)-1] = np.array([-0.8, -0.8]) # useful for BAColorGV

    elif config.dataset.dataset_name in ["MNIST", "CPatchMNIST", "CPatchMNIST2"]:
        pos = [ (x[4], -x[3])  for x in node_attr]
    elif "SST2" in config.dataset.dataset_name:
        pos = {i: (i*10, (-1)**(i)) for i in range(G.number_of_nodes())}
    
    node_colors = []
    for i in range(len(node_attr)):
        if len(node_gt) > 0 and node_gt[i]:
            node_colors.append("orange") # "lightgreen"
        elif len(node_gt) > 0 and not node_gt[i]:
            node_colors.append("blue") # "lightgreen"
        else:
            node_colors.append(get_color_based_on_dataset(config, node_attr[i]))

    nx.draw(
        G,
        with_labels=with_labels,
        pos=pos,
        ax=ax,
        node_size=nodesize,
        node_color=node_colors,
        edge_color=(0.55, 0.55, 0.55),
        alpha=0.9 if config.dataset.dataset_name in ["MNIST", "CPatchMNIST", "CPatchMNIST2"] else 0.5
    )


    if config.dataset.dataset_name != "MUTAG" and config.dataset.dataset_name != "GraphSST2Planted":
        if node_expl is not None:
            if thrs is not None:
                node_labels = {u: "E" if score >= thrs else "" for u, score in enumerate(node_expl)}
            else:
                node_labels = {u: "E" if is_relev else "" for u, is_relev in enumerate(list(nx.get_node_attributes(G, "node_mask").values()))}
        else:
            assert False, "Not implemented"
            edge_color = list(nx.get_edge_attributes(G, "attn_weight").values())
            edge_color = ["red" if e >= thrs else "black" for e in edge_color]
    elif config.dataset.dataset_name == "GraphSST2Planted":
        sentence_tokens = list(nx.get_node_attributes(G, "sentence_tokens").values())
        node_labels = {i: sentence_tokens[i] for i in range(len(node_attr))}
    else:
        index_to_symbol = {0: 'C', 1: 'O', 2: 'Cl', 3: "H", 4: "N", 5: "F", 6: "Br", 7: "S", 8: "P", 9: "I", 10: "Na", 11: "K", 12: "Li", 13: "Ca"}
        node_labels = {i: index_to_symbol[np.argmax(node_attr[i])] for i in range(len(node_attr))}

    # Annotate nodes with 'E' or other labels
    nx.draw_networkx_labels(
        G,
        pos,
        node_labels,
        font_size=12,
        font_color="red" if config.dataset.dataset_name in ["MNIST", "CPatchMNIST", "CPatchMNIST2"] else "black",
        alpha=0.6
    )

    # Annotate with edge scores
    # if nx.get_edge_attributes(G, 'attn_weight') != {}:
    #     nx.draw_networkx_edge_labels(
    #         G,
    #         pos,
    #         edge_labels=nx.get_edge_attributes(G, 'attn_weight'),
    #         font_size=6,
    #         alpha=0.8
    #     )

    # Annotate with node scores
    if node_expl is not None and pos is not None:
        offset = 0.05
        if isinstance(pos, dict):
            label_pos = {n: (x, y + offset) for n, (x, y) in pos.items()}  # vertical offset
        else:
            label_pos = {n: (x, y + offset) for n, (x, y) in enumerate(pos)}  # vertical offset

        nx.draw_networkx_labels(
            G,
            label_pos,
            labels={n: f"{v:.2f}" for n, v in enumerate(node_expl)},
            font_size=6,
            alpha=0.8
        )

    highlight_nodes = [u for u, is_relev in enumerate(list(nx.get_node_attributes(G, "node_mask").values())) if is_relev]
    nx.draw_networkx_nodes(
        G, 
        pos,
        nodelist=highlight_nodes,
        node_color="yellow",
        node_size=1000,   # larger than the actual node
        alpha=0.3         # transparent halo
    )

    # if topk is not None:
    #     ax = plt.gca()
    #     for node in topk:
    #         x, y = pos[node]
    #         circle = Circle((x, y), radius=0.05, edgecolor='grey', facecolor='none', linewidth=1)
    #         ax.add_patch(circle)
    
    plt.suptitle(title)

    if save:
        path = f'{ROOT_DIR}/GOOD/kernel/pipelines/plots/{subfolder}/{config.load_split}_{config.util_model_dirname}_{config.random_seed}/'
        if not os.path.exists(path):
            try:
                os.makedirs(path)
            except Exception as e:
                print(e)
                exit(e)
        plt.savefig(f'{path}/{name}.pdf')
    else:
        plt.show()

    plt.close()
    return pos

def plot_sentence_graph(G, name, subfolder, config, title):
    """
    Plot a sentence graph (e.g., from GraphSST2).
    
    Args:
        G: PyG Data object with attributes:
           - sentence_tokens: list of strings
           - edge_index: [2, E] tensor
        node_importance: optional 1D array/tensor of length num_nodes
        edge_importance: optional 1D array/tensor of length num_edges
        highlight_threshold: nodes with importance >= threshold will be highlighted
    """
    tokens = list(G.sentence_tokens)
    num_nodes = len(tokens)
    edges = G.edge_index.t().tolist()

    node_importance = G.node_expl

    fig, ax = plt.subplots(figsize=(len(tokens) * 0.7, 2))
    ax.set_axis_off()

    # Horizontal positions
    x_coords = np.arange(num_nodes)
    y_coords = np.zeros(num_nodes)

    # Draw words + highlights
    for i, token in enumerate(tokens):
        if G.node_mask[i]:
            ax.add_patch(plt.Rectangle((x_coords[i] - 0.4, -0.2), 0.8, 0.4,
                                       color='yellow', alpha=0.3, zorder=0))
        ax.text(x_coords[i], 0, token, ha='center', va='center', fontsize=10)
        ax.text(x_coords[i], -1, round(node_importance[i].item(), 2), ha='center', va='center', fontsize=6)

    # Draw edges as arcs
    for idx, (src, tgt) in enumerate(edges):
        if src == tgt:  # skip self-loops
            continue
        # Ensure left-to-right arc
        if src > tgt:
            src, tgt = tgt, src
        x1, x2 = x_coords[src], x_coords[tgt]
        xm = (x1 + x2) / 2
        width = abs(x2 - x1)
        height = width / 2 + 1
        # imp = float(edge_importance[idx])
        arc = Arc((xm, 1), width=width, height=height,
                      theta1=0, theta2=180, color='black',
                      alpha=0.5, lw=0.5)
        ax.add_patch(arc)

    ax.set_xlim(-1, num_nodes)
    ax.set_ylim(-1, max(2, num_nodes/2))
    plt.suptitle(title)
    plt.tight_layout()
    path = f'{ROOT_DIR}/GOOD/kernel/pipelines/plots/{subfolder}/{config.load_split}_{config.util_model_dirname}_{config.random_seed}/'
    
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except Exception as e:
            print(e)
            exit(e)
    plt.savefig(f'{path}/{name}.pdf')
    plt.close()

def random_attach(S, T):
    # random attach frontier nodes in S and T

    S_frontier = list(filter(lambda x: nx.get_node_attributes(S,'frontier').get(x, False), S.nodes()))
    T_frontier = list(filter(lambda x: nx.get_node_attributes(T,'frontier').get(x, False), T.nodes()))

    ret = nx.union(S, T, rename=("", "T"))
    for n in S_frontier:
        # pick random node v in G_t_spu
        # add edge (u,v) and (v,u)
        idx = randint(0, len(T_frontier)-1)
        v = "T" + str(T_frontier[idx])

        # assert str(n) in ret.nodes() and v in ret.nodes()

        ret.add_edge(str(n), v) #, origin="added"
        ret.add_edge(v, str(n)) #, origin="added"
    return ret

def random_attach_no_target_frontier(S, T):
    # random attach frontier nodes in S and T
    # avoid selecting target nodes that are in the frontier
    
    edge_attrs = list(nx.get_edge_attributes(S, "edge_attr").values())
    edge_gts = list(nx.get_edge_attributes(S, "edge_gt").values())
    S_frontier = list(filter(lambda x: nx.get_node_attributes(S,'frontier').get(x, False), S.nodes()))
    
    if edge_gts != []:
        nx.set_edge_attributes(T, 0, "edge_gt") # mark every edge of target graph as not GT edge

    ret = nx.union(S, T, rename=("", "T"))
    for n in S_frontier:
        # pick random node v in G_t_spu
        # add edge (u,v) and (v,u)
        idx = randint(0, len(T.nodes()) - 1)
        v = "T" + str(list(T)[idx])
        # assert str(n) in ret.nodes() and v in ret.nodes()

        if edge_attrs != []:
            attr = randint(0, len(edge_attrs) - 1)
            ret.add_edge(str(n), v, edge_attr=edge_attrs[attr])
            ret.add_edge(v, str(n), edge_attr=edge_attrs[attr])
        elif edge_gts != []:
            if edge_gts == []:
                ret.add_edge(str(n), v, edge_gt=0)
                ret.add_edge(v, str(n), edge_gt=0)
            else:
                ret.add_edge(str(n), v, edge_gt=0)
                ret.add_edge(v, str(n), edge_gt=0)
        else:
            ret.add_edge(str(n), v)
            ret.add_edge(v, str(n))
    return ret

def expl_acc(expl, data, expl_weight=None):
    edge_gt = {(u.item(),v.item()): data.edge_gt[i] for i, (u,v) in enumerate(data.edge_index.T)} 
    edge_expl = set([(u.item(),v.item()) for u,v in expl.T])
    
    # tp = int(sum([edge_gt[(u.item(),v.item())] for u,v in expl.T]))
    # fp = int(sum([not edge_gt[(u.item(),v.item())] for u,v in expl.T]))
    # tn = int(sum([not (u.item(),v.item()) in edge_expl and not edge_gt[(u.item(),v.item())] for u,v in data.edge_index.T]))
    # fn = int(sum([not (u.item(),v.item()) in edge_expl and edge_gt[(u.item(),v.item())] for u,v in data.edge_index.T]))
    
    # acc = (tp + tn) / (tp + fp + tn + fn)
    # f1 = 2*tp / (2*tp + fp + fn)
    f1 = 0.0
    # assert (tp + fp + tn + fn) == len(edge_gt)

    wiou, den = 0, 1e-12
    for i, (u,v) in enumerate((data.edge_index.T)):
        u, v = u.item(), v.item()
        if edge_gt[(u,v)]:
            if (u,v) in edge_expl:
                wiou += expl_weight[i].item()
                den += expl_weight[i].item()
            else:
                den += expl_weight[i].item()
        elif (u,v) in edge_expl:
            den += expl_weight[i].item()
    wiou = wiou / den
    return round(wiou, 3), round(f1, 3)

def expl_acc_fast(expl, data, expl_weight=None, reference_intersection=None):
    """
        Works under the assumption that expl=edge_index of the entire graph.
        This is because the stability_detector analysis is done via WIOU, which
        is evaluated over the entire graph, without the need to split into different
        ratios.
    """
    f1 = 0.0

    if reference_intersection is None:
        mask = data.edge_gt == 1
    else:
        mask = reference_intersection

    intersection = torch.sum(expl_weight[mask])
    union        = torch.sum(expl_weight)
    wiou_fast = intersection / (union + 1e-10)
    return torch.round(wiou_fast, decimals=3).item(), f1

def expl_acc_super_fast(batch_data, batch_edge_score, reference_intersection):
    """
        Works WITH BATCH OF DATA AND under the assumption that expl=edge_index of the entire graph.
        This is because the stability_detector analysis is done via WIOU, which
        is evaluated over the entire graph, without the need to split into different
        ratios.

        reference_intersection: What to consider to be defined the intersection of WIoU.
                                It can be either the GT explanation, resulting in Plausibility WIoU,
                                or the previously predicted hard explanaiton, resulting in Stability WIoU.
    """
    intersection = scatter_sum(batch_edge_score * reference_intersection, batch_data.batch[batch_data.edge_index[0]])
    union        = scatter_sum(batch_edge_score, batch_data.batch[batch_data.edge_index[0]])
    wiou_super_fast = intersection / (union + 1e-10)
    return torch.round(wiou_super_fast, decimals=3)

def fidelity(graph, type):
    """
        Generate the perturbed sample according to Fidelity+ and Fidelity-.
        I.e., either remove the entire explanation, or the entire complement.
        Operationally, we keep the node induced subgraph of either relevant or irrelevant nodes.
    """
    if graph.node_mask.sum() == 0: # discard empty explanations
        return None
    
    has_edge_attr = "edge_attr" in graph.keys()
    has_node_is_spurious = "node_is_spurious" in graph.keys()
    
    if type == "fidm":
        # preserve the node induced subgraph of relevant edges
        nodes_to_keep = graph.node_mask
    elif type == "fidp":
        # preserve the node induced subgraph of IRrelevant edges
        nodes_to_keep = torch.logical_not(graph.node_mask)

    edge_index, edge_attr, edge_mask = subgraph(
        nodes_to_keep,
        graph.edge_index,
        edge_attr=graph.edge_attr if has_edge_attr else None,
        return_edge_mask=True,
        relabel_nodes=True,
        num_nodes=graph.x.shape[0]
    )
    return [
        Data(
            x=graph.x[nodes_to_keep],
            edge_index=edge_index,
            edge_attr=edge_attr,
            node_is_spurious=graph.node_is_spurious[nodes_to_keep] if has_node_is_spurious else None,
            y=graph.y,
            node_expl=graph.node_expl[nodes_to_keep],
            node_mask=graph.node_mask[nodes_to_keep],
            edge_mask=graph.edge_mask[edge_mask],
        )
    ]

def robust_fidelity(graph, type, p, expval_budget, inplace=False):
    """
        Generate the perturbed sample according to Robust Fidelity+ and Robust Fidelity-.
        I.e., remove random edges in a IID fashion.
        Partially inspired from https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/utils/dropout.html#dropout_edge

        For Necessity metrics: note that this function considers every edge outside of the complement node-induced subgraph as candidate edge 
        to be removed. Thus, not only the ones strictly inside the explanation node-induced subgraph. 
        For Sufficiency metrics: Specular
    """
    if graph.node_mask.sum() == 0: # discard empty explanations
        return None
    
    has_edge_attr= "edge_attr" in graph.keys()
    has_node_is_spurious = "node_is_spurious" in graph.keys()
    
    if type == "rfidm":
        # sample IID for each edge, then force edges inside of R to remain
        nodes_to_keep = graph.node_mask
    elif type == "rfidp":
        # sample IID from the explanation, so get the subgraph induced by the complement
        nodes_to_keep = torch.logical_not(graph.node_mask)

    row, col = graph.edge_index
    _, _, force_to_keep = subgraph(
        nodes_to_keep,
        graph.edge_index,
        return_edge_mask=True,
        relabel_nodes=True,
        num_nodes=graph.x.shape[0]
    )   

    if inplace:
        ret = [graph]
    else:
        ret = [
            Data(
                x=graph.x,
                # edge_index=edge_index,
                edge_attr=None, #graph.edge_attr[idx_kept_edges] if has_edge_attr else None,
                node_is_spurious=graph.node_is_spurious if has_node_is_spurious else None,
                y=graph.y,
                node_expl=graph.node_expl,
                node_mask=graph.node_mask,
                # edge_mask=graph.edge_mask[idx_kept_edges],
            )
            for _ in range(expval_budget)
        ]

    edge_masks = torch.rand((expval_budget, row.size(0)), device=graph.edge_index.device) >= p
    edge_masks[:, force_to_keep] = True # force to keep edges inside R or C, based on the metric
    edge_masks[:, row > col] = False  # force undirected
    all_nonzero = edge_masks.nonzero()
    for j in range(expval_budget):
        edge_mask = edge_masks[j]
        edge_index = graph.edge_index[:, edge_mask]
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        
        # idx_kept_edges = edge_mask.nonzero().repeat((2, 1)).squeeze()
        # assert torch.all(edge_mask.nonzero() == all_nonzero[all_nonzero[:, 0] == j][:, 1].reshape(-1, 1))
        # idx_kept_edges_old = all_nonzero[all_nonzero[:, 0] == j][:, 1].reshape(-1, 1).repeat((2, 1)).squeeze()     

        idx_kept_edges = all_nonzero[all_nonzero[:, 0] == j][:, 1].repeat(2).squeeze()     
        
        ret[j].edge_index=edge_index
        ret[j].edge_mask=graph.edge_mask[idx_kept_edges]
        if has_edge_attr:
            ret[j].edge_attr=graph.edge_attr[idx_kept_edges]
    return ret

def nec_budget(graph, avg_graph_size, p, expval_budget):
    """
        Modification of RFID+ to account for irrelevant edges in the explanation.
        From 'https://openreview.net/pdf?id=kiOxNsrpQy'
        Instead of sampling edges IID, sample a fixed budget proportional to the average size of graphs.
        
        Note that this function considers every edge outside of the complement node-induced subgraph as candidate edge 
        to be removed. Thus, not only the ones strictly inside the explanation node-induced subgraph.
    """
    if graph.node_mask.sum() == 0: # discard empty explanations
        return None
    
    has_edge_attr= "edge_attr" in graph.keys()
    has_node_is_spurious = "node_is_spurious" in graph.keys()

    row, col = graph.edge_index
    complement_edge_index, _, force_to_keep_complement = subgraph(
        torch.logical_not(graph.node_mask),
        graph.edge_index,
        return_edge_mask=True,
        relabel_nodes=True,
        num_nodes=graph.x.shape[0]
    )

    ret = [
            Data(
                x=graph.x,
                # edge_index=edge_index,
                edge_attr=None, #graph.edge_attr[idx_kept_edges] if has_edge_attr else None,
                node_is_spurious=graph.node_is_spurious if has_node_is_spurious else None,
                y=graph.y,
                node_expl=graph.node_expl,
                node_mask=graph.node_mask,
                # edge_mask=graph.edge_mask[idx_kept_edges],
        )
        for _ in range(expval_budget)
    ] 
    
    # set to False (hence remove) the B edges with highest random weight
    B = min(int(p * avg_graph_size), (graph.edge_index.shape[1] - complement_edge_index.shape[1]))
    
    edge_weights = torch.rand((expval_budget, row.size(0)), device=graph.edge_index.device)
    edge_weights[:, force_to_keep_complement] = -torch.inf # make sure edges in C cannot be chosen
    edge_weights[:, row > col] = -torch.inf # force undirected while ensuring that exactly B edges are removed
    edges_to_remove = torch.topk(edge_weights, k=B, dim=1).indices # B edges with highest value are chosen to be removed

    edges_to_keep = torch.ones_like(edge_weights, device=graph.edge_index.device)
    edges_to_keep.scatter_(index=edges_to_remove, dim=1, value=False)
    edges_to_keep[:, row > col] = False  # force undirected
    edges_to_keep = edges_to_keep.bool()
    
    all_nonzero = edges_to_keep.nonzero()
    for j in range(expval_budget):
        edge_mask = edges_to_keep[j]

        edge_index = graph.edge_index[:, edge_mask]
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

        idx_kept_edges = all_nonzero[all_nonzero[:, 0] == j][:, 1].repeat(2).squeeze()
        
        ret[j].edge_index=edge_index
        ret[j].edge_mask=graph.edge_mask[idx_kept_edges]
        if has_edge_attr:
            ret[j].edge_attr=graph.edge_attr[idx_kept_edges]
    return ret


def suff_intervent(graph, graph_database, graph_database_labels, expval_budget):
    """
        Interventional SUFF from 'https://openreview.net/pdf?id=kiOxNsrpQy'.
        Randomy attach R from G with C' of a random G'.
        The number of new random edges is the same as the number of edges that 
        were removed from G to R (preserve number of edges, but randomly connect R with C').
    """
    def merge_graphs_randomly(data1: Data, data2: Data, num_random_edges, has_node_is_spurious) -> Data:
        num_nodes_1 = data1.num_nodes
        num_nodes_2 = data2.num_nodes

        # Offset the edge index of the second graph
        edge_index2 = data2.edge_index + num_nodes_1

        # Concatenate edge indices
        merged_edge_index = torch.cat([data1.edge_index, edge_index2], dim=1)

        # Concatenate X
        merged_x = torch.cat([data1.x, data2.x], dim=0)        

        # Concatenate edge features (if available)
        if hasattr(data1, 'edge_attr') and data1.edge_attr is not None:
            merged_edge_attr = torch.cat([data1.edge_attr, data2.edge_attr], dim=0)
        else:
            merged_edge_attr = None

        # Add random edges between the two graphs (avoiding duplicates)
        if num_random_edges > 0:
            all_possible_edges = torch.cartesian_prod(
                torch.arange(num_nodes_1),
                torch.arange(start=num_nodes_1, end=num_nodes_1+num_nodes_2)
            ).to(data1.y.device)

            if has_node_is_spurious:
                # remove edges connecting G/V
                all_possible_edges_is_spurious = torch.cartesian_prod(
                    data1.node_is_spurious,
                    data2.node_is_spurious
                ).to(data1.y.device)
                all_possible_edges = all_possible_edges[all_possible_edges_is_spurious.sum(1) == 0,:]
            random_edges_idxs = torch.randperm(all_possible_edges.shape[0])[:num_random_edges]
            random_edges = all_possible_edges[random_edges_idxs, :].T # size: 2xnum_random_edges

            # Make bidirectional
            bidir_edges = torch.cat([random_edges, random_edges[[1, 0]]], dim=1)
            merged_edge_index = torch.cat([merged_edge_index, bidir_edges], dim=1)

            if merged_edge_attr is not None:
                rand_edge_attr = torch.zeros((bidir_edges.size(1), merged_edge_attr.size(1)), device=data1.y.device)
                # rand_edge_mask = torch.zeros((bidir_edges.size(1), merged_edge_attr.size(1)), device=data1.y.device) # CHECK IF IT IS NEEDED
                merged_edge_attr = torch.cat([merged_edge_attr, rand_edge_attr], dim=0)

        return Data(
            x=merged_x,
            edge_index=merged_edge_index,
            edge_attr=merged_edge_attr,
            node_is_spurious=torch.cat([data1.node_is_spurious, data2.node_is_spurious], dim=0) if has_node_is_spurious else None,
            y=data1.y, # Watch out! this holds only in the invariance setup
            node_expl=torch.cat([data1.node_expl, data2.node_expl], dim=0),
            node_mask=torch.cat([data1.node_mask, data2.node_mask], dim=0),
            edge_mask=torch.cat([data1.edge_mask, data2.edge_mask], dim=0),
        )
    
    def count_boundary_edges(edge_index: torch.Tensor, subset: torch.Tensor, num_nodes: int) -> int:
        # Create a mask for nodes in the subset
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        mask[subset] = True

        src, dst = edge_index

        # An edge is a boundary edge if one end is in subset and the other is not
        in_subset = mask[src]
        in_complement = ~mask[dst]

        # Forward direction: src in subset, dst outside
        boundary_forward = in_subset & in_complement

        # Reverse direction: dst in subset, src outside (for undirected graphs)
        in_subset_rev = mask[dst]
        in_complement_rev = ~mask[src]
        boundary_backward = in_subset_rev & in_complement_rev

        # Combine both directions
        boundary_edges = boundary_forward | boundary_backward
        return boundary_edges.sum().item() // 2 # count only one direction
    
    if graph.node_mask.sum() == 0: # discard empty explanations
        return None
    
    has_node_is_spurious = "node_is_spurious" in graph.keys()
    
    # Construct the Data object for R of G
    edge_index, edge_attr, edge_mask = subgraph(
        graph.node_mask,
        graph.edge_index,
        edge_attr=graph.edge_attr if "edge_attr" in graph.keys() else None,
        return_edge_mask=True,
        relabel_nodes=True,
        num_nodes=graph.x.shape[0]
    )
    R = Data(
        x=graph.x[graph.node_mask],
        edge_index=edge_index,
        edge_attr=edge_attr,
        node_is_spurious=graph.node_is_spurious[graph.node_mask] if has_node_is_spurious else None,
        y=graph.y,
        node_expl=graph.node_expl[graph.node_mask],
        node_mask=graph.node_mask[graph.node_mask],
        edge_mask=graph.edge_mask[edge_mask],
    )

    ret = []
    same_class_idx = (graph_database_labels == graph.y.item()).nonzero(as_tuple=True)[0]
    rnd_idxs = torch.randint(0, len(same_class_idx), (expval_budget,))
    num_random_edges = count_boundary_edges(edge_index=graph.edge_index, subset=graph.node_mask, num_nodes=graph.x.size(0))
    for i in range(expval_budget):
        # Sample G' from ANY class
        # Suitable only where the subgraph invariance holds
        # graph_to_merge = graph_database[randint(0, len(graph_database) - 1)]

        # Sample G' from same class as G
        rand_idx = same_class_idx[rnd_idxs[i].item()]
        graph_to_merge = graph_database[rand_idx]

        # Construct the Data object for C' of G'
        edge_index, edge_attr, edge_mask = subgraph(
            torch.logical_not(graph_to_merge.node_mask),
            graph_to_merge.edge_index,
            edge_attr=graph_to_merge.edge_attr if "edge_attr" in graph_to_merge.keys() else None,
            return_edge_mask=True,
            relabel_nodes=True,
            num_nodes=graph_to_merge.x.shape[0]
        )
        C_dash = Data(
            x=graph_to_merge.x[torch.logical_not(graph_to_merge.node_mask)],
            edge_index=edge_index,
            edge_attr=edge_attr,
            node_is_spurious=graph_to_merge.node_is_spurious[torch.logical_not(graph_to_merge.node_mask)] if has_node_is_spurious else None,
            y=graph_to_merge.y,
            node_expl=graph_to_merge.node_expl[torch.logical_not(graph_to_merge.node_mask)],
            node_mask=graph_to_merge.node_mask[torch.logical_not(graph_to_merge.node_mask)],
            edge_mask=graph_to_merge.edge_mask[edge_mask],
        )
        # C_dash = Data(
        #     x=torch.tensor([[0., 0., 0., 9.]], device=graph.y.device),
        #     edge_index=torch.empty(2, 0, dtype=torch.long, device=graph.y.device),
        #     node_is_spurious=torch.tensor([1], device=graph.y.device),
        #     edge_attr=None,
        #     y=None,
        #     node_expl=torch.tensor([9], device=graph.y.device),
        #     node_mask=torch.tensor([9], device=graph.y.device),
        #     edge_mask=torch.tensor([9], device=graph.y.device),
        # )
        
        # Merge R with C'
        ret.append(
            merge_graphs_randomly(R, C_dash, num_random_edges, has_node_is_spurious)
        )
    return ret


def counter_fid(graph, expval_budget):
    """
        Implementation of Counterfacual Fidelity as described in Alg. 1 of 'https://arxiv.org/pdf/2406.07955'.
        Samples random explanation scores with mean and std as given by the explanatory scores of the input.
        The perturbed input has altered attention scores, and needs to be forwarded to the CLF only (line 12 of Alg. 1).
    """
    if graph.node_mask.sum() == 0: # discard empty explanations
        return None
    
    if "node_expl" in graph.keys() and not "edge_expl" in graph.keys():
        mean_attn_scores = torch.mean(graph.node_expl)
        std_attn_scores = torch.std(graph.node_expl) + 1e-6
        if torch.isnan(std_attn_scores):
            std_attn_scores = 1e-6
    elif "edge_expl" in graph.keys() and not "node_expl" in graph.keys():
        raise ValueError("edge level explanation not supported")
    else:
        raise ValueError("configuration not suported")
    
    ret = []
    normal_dist = torch.distributions.Normal(loc=mean_attn_scores, scale=std_attn_scores)
    for _ in range(expval_budget):
        ret.append(graph.clone())
        ret[-1].node_expl = normal_dist.sample((graph.node_expl.size(0),)).to(graph.node_expl.device).sigmoid()
    return ret


def suff_cause(graph, expval_budget):
    """
        Our proposed SUFF.
        Remove both nodes and edges, at random.
        First subsample nodes at random. Then, remove edges at random by relying on RFID-.
    """
    if graph.node_mask.sum() == 0: # discard empty explanations
        return None
    
    has_edge_attr= "edge_attr" in graph.keys()
    has_node_is_spurious = "node_is_spurious" in graph.keys()
    
    ret = []
    rnd_weights = torch.rand((expval_budget, graph.x.shape[0]), device=graph.x.device)
    rnd_weights[:, graph.node_mask] = 1.0 # always keep nodes in R
    nodes_to_keep_mask = rnd_weights >= 0.5 # keep nodes with a score >= 0.5 (thus R + other random nodes)
    for i in range(expval_budget):
        # rnd_weights = torch.rand(graph.x.shape[0], device=graph.x.device)
        # rnd_weights[graph.node_mask] = 1.0 # always keep nodes in R
        # nodes_to_keep_mask = rnd_weights >= 0.5 # keep nodes with a score >= 0.5 (thus R + other random nodes)
        # nodes_to_keep = torch.arange(graph.x.shape[0])[nodes_to_keep_mask[i]]
        nodes_to_keep = torch.nonzero(nodes_to_keep_mask[i]).view(-1)

        edge_index, edge_attr, edge_mask = subgraph(
            nodes_to_keep,
            graph.edge_index,
            edge_attr=graph.edge_attr if has_edge_attr else None,
            return_edge_mask=True,
            relabel_nodes=True,
            num_nodes=graph.x.shape[0]
        )

        graph_node_sampled = Data(
            x=graph.x[nodes_to_keep],
            edge_index=edge_index,
            edge_attr=edge_attr,
            node_is_spurious=graph.node_is_spurious[nodes_to_keep] if has_node_is_spurious else None,
            y=graph.y,
            node_expl=graph.node_expl[nodes_to_keep],
            node_mask=graph.node_mask[nodes_to_keep],
            edge_mask=graph.edge_mask[edge_mask],
        )
        graph_node_edge_sampled = robust_fidelity(
            graph_node_sampled,
            type="rfidm",
            p=0.5,
            expval_budget=1,
            inplace=True
        )[0]
        ret.append(graph_node_edge_sampled)
    return ret

def suff_cause_soft(graph, expval_budget):
    """
        Soft version of SUFF_Cause.
        It simply set to zero the nodea features of nodes to be removed
    """
    if graph.node_mask.sum() == 0: # discard empty explanations
        return None
    
    ret = []
    rnd_weights = torch.rand((expval_budget, graph.x.shape[0]), device=graph.x.device)
    rnd_weights[:, graph.node_mask] = 1.0 # always keep nodes in R
    nodes_to_keep_mask = rnd_weights >= 0.5 # keep nodes with a score >= 0.5 (thus R + other random nodes)
    for i in range(expval_budget):
        nodes_to_remove = torch.nonzero(torch.logical_not(nodes_to_keep_mask[i])).view(-1)
        graph_node_sampled = graph.clone()
        # graph_node_sampled.x *= nodes_to_keep_mask[i].unsqueeze(1)
        graph_node_sampled.x[nodes_to_remove, :] = 0

        # graph_node_sampled = Data(
        #     x=graph.x[nodes_to_keep],
        #     edge_index=edge_index,
        #     edge_attr=edge_attr,
        #     node_is_spurious=graph.node_is_spurious[nodes_to_keep] if has_node_is_spurious else None,
        #     y=graph.y,
        #     node_expl=graph.node_expl[nodes_to_keep],
        #     node_mask=graph.node_mask[nodes_to_keep],
        #     edge_mask=graph.edge_mask[edge_mask],
        # )
        # graph_node_edge_sampled = robust_fidelity(
        #     graph_node_sampled,
        #     type="rfidm",
        #     p=0.5,
        #     expval_budget=1,
        #     inplace=True
        # )[0]
        # ret.append(graph_node_edge_sampled)
        ret.append(graph_node_sampled)
    return ret


def sample_edges_tensorized(data, nec_number_samples, sampling_type, nec_alpha_1, avg_graph_size, edge_index_to_remove=None, force_undirected=True):
    if sampling_type == "bernoulli":
        assert not nec_alpha_1 is None
        # customization of dropout_edge from https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/utils/dropout.html#dropout_edge
        data = data.clone()

        row, col = data.edge_index
        mask_noncausal = torch.ones(row.size(0), dtype=bool)
        mask_noncausal[edge_index_to_remove] = False
        
        edge_mask = torch.rand(row.size(0), device=data.edge_index.device) >= nec_alpha_1
        edge_mask[mask_noncausal] = True
        if force_undirected:
            edge_mask[row > col] = False

        edge_index = data.edge_index[:, edge_mask]

        if force_undirected:
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
            edge_id_kept = edge_mask.nonzero().repeat((2, 1)).squeeze()
        else:
            edge_id_kept = edge_mask

        if hasattr(data, "edge_attr") and not data.edge_attr is None:
            data.edge_attr = data.edge_attr[edge_id_kept]
        
        data.edge_index, data.edge_attr, mask = remove_isolated_nodes(edge_index, data.edge_attr, num_nodes=data.x.shape[0])
        data.x = data.x[mask]
        data.num_nodes = data.x.shape[0]
        return data
    elif sampling_type == "deconfounded":
        if nec_number_samples == "prop_G_dataset":
            k = max(1, int(nec_alpha_1 * avg_graph_size))
        elif nec_number_samples == "prop_R":
            k = max(1, int(nec_alpha_1 * edge_index_to_remove.sum()))
            # if k == max(1, int(nec_alpha_1 * avg_graph_size)):
            #     print(k)
        elif nec_number_samples == "alwaysK":
            k = nec_alpha_1
        else:
            raise ValueError(f"value for nec_number_samples ({nec_number_samples}) not supported")
        
        k = min(k, int(data.edge_index.shape[1]/2)-2)
        assert k > 0, k

        data = data.clone()
        row, col = data.edge_index
        undirected = data.edge_index[:, row <= col]

        candidate_mask = edge_index_to_remove[row <= col]
        candidate_idxs = torch.argwhere(candidate_mask)
        
        # Version of the main paper with permutation (requires for-loop)
        perm = torch.randperm(candidate_idxs.shape[0])
        to_keep = perm[:-k]
        removed = perm[-k:]

        causal_idxs_keep = candidate_idxs[to_keep].view(-1)
        causal_idxs_remove = candidate_idxs[removed].view(-1)

        to_keep = torch.zeros(undirected.shape[1], dtype=torch.bool)
        to_keep[candidate_mask == 0] = 1
        to_keep[causal_idxs_keep] = 1

        data.edge_index = torch.cat((undirected[:, to_keep], undirected[:, to_keep].flip(0)), dim=1)
        
        if hasattr(data, "edge_attr") and not data.edge_attr is None:
            undirected_edge_attr = data.edge_attr[row <= col]
            data.edge_attr = torch.cat((undirected_edge_attr[to_keep, :], undirected_edge_attr[to_keep, :]), dim=0)
        if hasattr(data, "edge_gt"):            
            undirected_edge_gt = data.edge_gt[row <= col]
            data.edge_gt = torch.cat((undirected_edge_gt[to_keep], undirected_edge_gt[to_keep]), dim=0)
        if hasattr(data, "causal_mask"):
            undirected_causal_mask = data.causal_mask[row <= col]
            data.causal_mask = torch.cat((undirected_causal_mask[to_keep], undirected_causal_mask[to_keep]), dim=0)

        data.edge_index, data.edge_attr, mask = remove_isolated_nodes(data.edge_index, data.edge_attr, num_nodes=data.x.shape[0])
        data.x = data.x[mask]
        data.num_nodes = data.x.shape[0]
        return data
    else:
        raise ValueError(f"sampling_type {sampling_type} not valid")
    

def sample_edges_tensorized_batched(
        data,
        nec_number_samples,
        sampling_type, 
        nec_alpha_1,
        avg_graph_size,
        budget,
        edge_index_to_remove=None,
):
    if sampling_type == "bernoulli":
        raise NotImplementedError("")
    elif sampling_type == "deconfounded":
        if nec_number_samples == "prop_G_dataset":
            k = max(1, int(nec_alpha_1 * avg_graph_size))
        elif nec_number_samples == "prop_R":
            k = max(1, int(nec_alpha_1 * edge_index_to_remove.sum()))
        elif nec_number_samples == "alwaysK":
            k = nec_alpha_1
        else:
            raise ValueError(f"value for nec_number_samples ({nec_number_samples}) not supported")
        
        row, col = data.edge_index
        undirected = data.edge_index[:, row <= col]

        candidate_mask = edge_index_to_remove[row <= col]
        candidate_idxs = torch.argwhere(candidate_mask)
        
        k = min(k, int(data.edge_index.shape[1]/2)-2, candidate_idxs.shape[0])
        if k == 0:
            return None # None | [data.clone() for _ in range(budget)]

        # New version without perm, to avoid for loop
        random_weight_per_index = torch.rand(budget, candidate_idxs.shape[0], device=data.edge_index.device)
        topk_weight_per_index = torch.topk(random_weight_per_index, k=k, largest=True, dim=-1)
        
        all_except_topk = torch.ones(budget, candidate_idxs.shape[0], dtype=torch.bool)
        all_except_topk.scatter_(1, topk_weight_per_index.indices, False)

        to_keep = torch.arange(candidate_idxs.shape[0]).repeat(budget, 1)
        to_keep = to_keep.flatten()[all_except_topk.flatten()].reshape(to_keep.shape[0], all_except_topk.sum(-1)[0])
        # removed = topk_weight_per_index.indices

        causal_idxs_keep = candidate_idxs.reshape(1, -1).repeat(budget, 1).gather(1, to_keep) # B x elem_to_keep: indexes of edges to keep as elements
        # causal_idxs_remove = candidate_idxs.reshape(1, -1).repeat(budget, 1).gather(1, to_keep)

        to_keep = torch.zeros(budget, undirected.shape[1], dtype=torch.bool)
        to_keep[candidate_mask.repeat(budget, 1) == 0] = 1
        to_keep.scatter_(1, causal_idxs_keep, 1)

        intervened_graphs = []
        for k in range(budget):
            intervened_data = data.clone()
            intervened_data.edge_index = torch.cat((undirected[:, to_keep[k]], undirected[:, to_keep[k]].flip(0)), dim=1)
        
            if not (getattr(data, "edge_attr", None) is None):
                undirected_edge_attr = intervened_data.edge_attr[row <= col]
                intervened_data.edge_attr = torch.cat((undirected_edge_attr[to_keep[k], :], undirected_edge_attr[to_keep[k], :]), dim=0)
            if not (getattr(data, "edge_gt", None) is None):
                undirected_edge_gt = intervened_data.edge_gt[row <= col]
                intervened_data.edge_gt = torch.cat((undirected_edge_gt[to_keep[k]], undirected_edge_gt[to_keep[k]]), dim=0)
            if not (getattr(data, "causal_mask", None) is None):
                undirected_causal_mask = intervened_data.causal_mask[row <= col]
                intervened_data.causal_mask = torch.cat((undirected_causal_mask[to_keep[k]], undirected_causal_mask[to_keep[k]]), dim=0)

            # intervened_data.edge_index, intervened_data.edge_attr, mask = remove_isolated_nodes(
            #     intervened_data.edge_index,
            #     intervened_data.edge_attr,
            #     num_nodes=intervened_data.x.shape[0]
            # )
            # if (~mask).sum() > 0: # at least one node was removed
            #     assert intervened_data.edge_index.shape[1] ==  intervened_data.edge_gt.shape[0], f"shape mismatch after remove_isolated_nodes(): {intervened_data.edge_index.shape[1]} vs {intervened_data.edge_gt.shape[0]}"
            # intervened_data.x = intervened_data.x[mask]
            # intervened_data.node_gt = intervened_data.node_gt[mask]
            # intervened_data.num_nodes = intervened_data.x.shape[0]
            intervened_data.num_edge_removed = data.edge_index.shape[1] - intervened_data.edge_index.shape[1]
            intervened_graphs.append(intervened_data)
        return intervened_graphs
    else:
        raise ValueError(f"sampling_type {sampling_type} not valid")

        

def explanation_stability_hard(data, explanations, ratio):
    """
        Minimal example of the pitfall of F1:
        >>> from sklearn.metrics import f1_score, matthews_corrcoef
        >>> import numpy as np
        >>> true = np.array([0,1,1,1])
        >>> pred = np.array([1,1,1,1])
        >>> f1_score(true, pred, pos_label=1)
            0.8571428571428571
        >>> f1_score(true, pred, pos_label=0)
            0.0
        >>> matthews_corrcoef(true, pred)
            0.0
    """
    # Extract new hard explanation
    (causal_edge_index, _, _, causal_batch), \
        _, mask_batch = split_graph(
            data,
            explanations,
            ratio,
            return_batch=True,
            compensate_edge_removal=data.num_edge_removed
    )     

    # (Slow version)
    # f1_single = []
    # for j in range(causal_batch.max() + 1):
    #     original_causal_mask = data.causal_mask[data.batch[data.edge_index[0]] == j].cpu()
    #     intervened_causal_mask = mask[data.batch[data.edge_index[0]] == j].cpu()
    #     f1_single.append(f1_score(original_causal_mask, intervened_causal_mask))

    # Calculating precision, recall, and F1 score using PyTorch (basic 1D case)
    # TP = ((input == 1) & (target == 1)).sum().item()
    # FP = ((input == 1) & (target == 0)).sum().item()
    # FN = ((input == 0) & (target == 1)).sum().item()

    # (Fast version)
    eps = 1e-6
    input = [] # mask
    target = [] # data.causal_mask

    # Make mask and causal_mask both undirected. Othewrise there could be a mismatch since split_graph()
    # does not always return both directionalities for a certain edge, while for intervened graphs 
    # causal_mask is forced to do so.
    for j, d in enumerate(data.to_data_list()):
        mask = mask_batch[data.batch[data.edge_index[0]] == j]
        row, col = d.edge_index
        undirected_mask = mask[row < col]
        new_mask = torch.cat((undirected_mask, undirected_mask), dim=0)
        causal_mask = torch.cat((d.causal_mask[row < col], d.causal_mask[row < col]), dim=0)

        # Manually add individual self-loops that, being directed, are excluded from above
        new_mask = torch.cat((new_mask, mask[row == col]), dim=0)
        causal_mask = torch.cat((causal_mask, d.causal_mask[row == col]), dim=0)

        input.append(new_mask)
        target.append(causal_mask)

    input = torch.cat(input, dim=0)
    target = torch.cat(target, dim=0)

    # TODO: for interventions adding elements, manually add novel edges to the counts
    TP = scatter_add((input & target).to(int), data.batch[data.edge_index[0]], dim=0)
    TN = scatter_add(((input == False) & (target == False)).to(int), data.batch[data.edge_index[0]], dim=0)
    FP = scatter_add(((input == True) & (target == False)).to(int), data.batch[data.edge_index[0]], dim=0)
    FN = scatter_add(((input == False) & (target == True)).to(int), data.batch[data.edge_index[0]], dim=0)

    # Compute F1
    precision = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)
    f1_batched = 2 * (precision * recall) / (precision + recall + eps)

    # Compute MCC
    numerator = (TP * TN) - (FP * FN)
    denominator = torch.sqrt(((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)))
    mcc_batched = numerator / (denominator + eps)
    return mcc_batched, f1_batched


def feature_intervention(G, feature_bank, feat_int_alpha):
    """Randomly swap feature of spurious graph with features sampled from a fixed bank"""
    assert feature_bank .shape[0] > 0

    G = G.copy()
    probs = np.random.binomial(1, feat_int_alpha, len(G))
    for i, n in enumerate(G):
        if probs[i] == 1:
            new_feature = feature_bank[randint(0, feature_bank.shape[0]-1)].tolist()
            nx.set_node_attributes(G, {n: new_feature}, name="ori_x")
    return G
