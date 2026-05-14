r"""
The implementation of `Discovering Invariant Rationales for Graph Neural Networks <https://openreview.net/pdf?id=hGXij5rfiHw>`_.
"""

import copy
import math

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import BatchNorm
from torch_geometric.data import Data
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import degree

from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from GOOD.utils.train import lift_node_att_to_edge_att

from .BaseGNN import GNNBasic
from .GINs import FeatExtractor
from .Classifiers import Classifier
from torch_geometric.utils.loop import add_self_loops, remove_self_loops
from torch_geometric.utils import is_undirected, to_undirected, coalesce, subgraph
from torch_sparse import transpose
from torch_geometric import __version__ as __pyg_version__

import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx



@register.model_register
class DIR(GNNBasic):

    def __init__(self, config: Union[CommonArgs, Munch]):
        super(DIR, self).__init__(config)

        self.att_net = CausalAttNet(config.ood.ood_param, config)
        
        if config.mitigation_sampling == "raw":
            print("Init CLASSIFIER")
            fe_kwargs = {'mitigation_readout': config.mitigation_readout}
            fe_kwargs["gnn_clf_layer"] = config.model.gnn_clf_layer
            fe_kwargs["no_bias"] = True

            self.gnn_clf = FeatExtractor(config, **fe_kwargs)
            print(f"Using mitigation_sampling==raw with {config.model.gnn_clf_layer} gnn_clf_layers")
        else:
            config.model.model_layer = config.model.model_layer - 2
            self.gnn_clf = FeatExtractor(config, without_embed=True)

        output_dim = 2
        if config.dataset.num_classes > 2:
            output_dim = config.dataset.num_classes

        self.learn_edge_att = config.ood.extra_param[0]
        self.classifierS = Classifier(config, output_dim=output_dim, is_linear=False)
        self.conf_classifierS = Classifier(config, output_dim=output_dim, is_linear=False)
        self.edge_mask = None

    def forward(self, *args, **kwargs):
        r"""
        The DIR model implementation.

        Args:
            *args (list): argument list for the use of arguments_read. Refer to :func:`arguments_read <GOOD.networks.models.BaseGNN.GNNBasic.arguments_read>`
            **kwargs (dict): key word arguments for the use of arguments_read. Refer to :func:`arguments_read <GOOD.networks.models.BaseGNN.GNNBasic.arguments_read>`

        Returns (Tensor):
            Label predictions and other results for loss calculations.

        """
        data = kwargs.get('data')
        batch_size = data.batch[-1].item() + 1

        (causal_x, causal_edge_index, causal_edge_attr, causal_node_weight, causal_edge_weight, causal_batch), \
        (conf_x, conf_edge_index, conf_edge_attr, conf_node_weight, conf_edge_weight, conf_batch), \
            (node_att_logit, node_att) = self.att_net(*args, **kwargs)

        # --- Causal repr ---
        set_masks(causal_edge_weight, self, causal_node_weight)
        causal_rep = self.get_graph_rep(
            data=Data(x=causal_x, edge_index=causal_edge_index,
                      edge_attr=causal_edge_attr, batch=causal_batch),
            batch_size=batch_size
        )
        causal_out = self.get_causal_pred(causal_rep)
        clear_masks(self)

        self.edge_mask = causal_edge_weight

        # if self.training:
            # --- Conf repr ---
            
        set_masks(conf_edge_weight, self, conf_node_weight)
        conf_rep = self.get_graph_rep(
            data=Data(x=conf_x, edge_index=conf_edge_index,
                        edge_attr=conf_edge_attr, batch=conf_batch),
            batch_size=batch_size
        ).detach()
        conf_out = self.get_conf_pred(conf_rep)
        clear_masks(self)

        # --- combine to causal phase (detach the conf phase) ---
        rep_out = None
        # rep_out = []
        # for idx, conf in enumerate(conf_rep):
        #     rep_out.append(self.get_comb_pred(causal_rep, conf))
        # rep_out = torch.stack(rep_out, dim=0)
        # DEBUG EFFICIENT VERSION
        # assert torch.allclose(rep_out, rep_out2, atol=1e-6)

        # --- combine to causal phase (Optimized version) ---
        rep_out2 = torch.transpose(
            self.get_comb_pred_eff(causal_rep, conf_rep),
            0,
            1
        ) # rep_out2[i] contains a tensor with every causal_rep keeping fixed conf_rep[i]
        
        # print()
        # print("causal_x:", causal_x[causal_batch==0].sum(0), causal_x[causal_batch==1].sum(0))
        # print("conf_x:", conf_x[conf_batch==0].sum(0), conf_x[conf_batch==1].sum(0))
        # print()
        # print("causal_out:", causal_out, causal_out.sigmoid())
        # print("conf_out:", conf_out, conf_out.sigmoid())
        # print("rep_out2:", rep_out2, rep_out2.sigmoid())


        # targets = data.y 
        # tmp = self.config.metric.loss_func(
        #     rep_out2.reshape(rep_out2.shape[0] * rep_out2.shape[0], 2), 
        #     targets.expand(targets.shape[0], -1, 2).reshape(rep_out2.shape[0] * rep_out2.shape[0], 2),
        #     reduction='none'
        # )            
        # print()
        # tmp = tmp.reshape(rep_out2.shape[0], rep_out2.shape[0])
        # tmp = tmp.mean(-1)
        # print("mean_loss:", tmp.mean())
        # exit("deb")


        return (rep_out, rep_out2), causal_out, conf_out, (node_att_logit, node_att)
        # else:
            # return causal_out, node_att_logit, node_att

    def get_graph_rep(self, *args, **kwargs):
        return self.gnn_clf(*args, **kwargs)

    def get_causal_pred(self, h_graph):
        return self.classifierS(h_graph)

    def get_conf_pred(self, conf_graph_x):
        return self.conf_classifierS(conf_graph_x)

    def get_comb_pred(self, causal_graph_x, conf_graph_x):
        causal_pred = self.classifierS(causal_graph_x)
        conf_pred = self.conf_classifierS(conf_graph_x).detach()
        return torch.sigmoid(conf_pred) * causal_pred
    
    def get_comb_pred_eff(self, causal_graph_x, conf_graph_x):
        causal_pred = self.classifierS(causal_graph_x)
        conf_pred = self.conf_classifierS(conf_graph_x).detach()
        return torch.sigmoid(conf_pred).unsqueeze(0) * causal_pred.unsqueeze(1)
    
    @torch.no_grad()
    def predict_from_subgraph(self, edge_att=False, log=None, eval_kl=None, node_att=False, *args, **kwargs):
        # if self.gnn_clf:
        #     lc_logits = self.classifierS(self.gnn_clf(*args, **kwargs))
        # else:
        #     lc_logits = self.classifierS(self.gnn(*args, **kwargs))

        set_masks(edge_att, self, node_att)
        causal_rep = self.get_graph_rep(
            *args, **kwargs
        )
        lc_logits = self.get_causal_pred(causal_rep)
        clear_masks(self)

        if log is None:
            if lc_logits.shape[-1] > 1:
                return lc_logits.softmax(-1)
            else:
                return lc_logits.sigmoid()
        else:
            assert False
            assert not (eval_kl is None)
            if lc_logits.shape[-1] > 1:
                return lc_logits.log_softmax(dim=1)
            else:
                if eval_kl: # make the single logit a proper distribution summing to 1 to compute KL
                    lc_logits = lc_logits.sigmoid()
                    new_logits = torch.zeros((lc_logits.shape[0], lc_logits.shape[1]+1), device=lc_logits.device)
                    new_logits[:, 1] = new_logits[:, 1] + lc_logits.squeeze(1)
                    new_logits[:, 0] = 1 - new_logits[:, 1]
                    new_logits[new_logits == 0.] = 1e-10
                    return new_logits.log()
                else:
                    return lc_logits.sigmoid().log()
    
    @torch.no_grad()
    def get_subgraph(self, *args, **kwargs):
        (rep_out, rep_out2), causal_out, conf_out, (node_att_logit, node_att) = self.forward(*args, **kwargs)
        return self.edge_mask, node_att, causal_out
    
    @torch.no_grad()
    def probs(self, *args, **kwargs):
        (rep_out, rep_out2), causal_out, conf_out, (node_att_logit, node_att) = self(*args, **kwargs)

        if causal_out.shape[-1] > 1:
            return causal_out.softmax(dim=1)
        else:
            return causal_out.sigmoid()

@register.model_register
class DIRvGIN(DIR):
    r"""
    The GIN virtual node version of DIR.
    """

    def __init__(self, config: Union[CommonArgs, Munch]):
        super(DIRvGIN, self).__init__(config)
        assert False
        self.att_net = CausalAttNet(config.ood.ood_param, config, virtual_node=True)
        config_fe = copy.deepcopy(config)
        config_fe.model.model_layer = config.model.model_layer - 2
        self.feat_encoder = vFeatExtractor(config_fe, without_embed=True)

@register.model_register
class DIRvGINNB(DIR):
    r"""
    The GIN virtual node without batchnorm version of DIR.
    """

    def __init__(self, config: Union[CommonArgs, Munch]):
        super(DIRvGINNB, self).__init__(config)
        assert False
        self.att_net = CausalAttNet(config.ood.ood_param, config, virtual_node=True, no_bn=True)
        config_fe = copy.deepcopy(config)
        config_fe.model.model_layer = config.model.model_layer - 2
        self.feat_encoder = vFeatExtractor(config_fe, without_embed=True)


class CausalAttNet(nn.Module):
    r"""
    Causal Attention Network adapted from https://github.com/wuyxin/dir-gnn.
    """

    def __init__(self, causal_ratio, config, **kwargs):
        super(CausalAttNet, self).__init__()

        config_catt = copy.deepcopy(config)

        if kwargs.get('virtual_node'):
            assert False, "Virtual node not in use"
            self.gnn_node = vFeatExtractor(config_catt, without_readout=True, **kwargs)
        else:
            self.gnn_node = FeatExtractor(config_catt, without_readout=True, **kwargs)
        
        self.learn_edge_att = config.ood.extra_param[0]
        self.extractor = ExtractorMLP(config)
        self.ratio = causal_ratio
        self.config = config

        print("Causal ratio = ", self.ratio)

    def forward(self, *args, **kwargs):
        data = kwargs.get('data') or None

        x = self.gnn_node(*args, **kwargs)

        att_log_logits = self.extractor(x, data.edge_index, data.batch)
        att = att_log_logits.sigmoid() # WARNING: Not in the original implementation

        if data.edge_index.shape[1] != 0:
            if self.learn_edge_att:
                if is_undirected(data.edge_index):
                    if self.config.average_edge_attn == "default":
                        nodesize = data.x.shape[0]
                        edge_att = (att + transpose(data.edge_index, att, nodesize, nodesize, coalesced=False)[1]) / 2
                    else:
                        data.ori_edge_index = data.edge_index.detach().clone() #for backup and debug
                        data.edge_index, edge_att = to_undirected(data.edge_index, att.squeeze(-1), reduce="mean")

                        if not data.edge_attr is None:
                            edge_index_sorted, edge_attr_sorted = coalesce(data.ori_edge_index, data.edge_attr, is_sorted=False)                    
                            data.edge_attr = edge_attr_sorted    
                else:
                    edge_att = att
                    
                (causal_edge_index, causal_edge_attr, causal_edge_weight), \
                    (conf_edge_index, conf_edge_attr, conf_edge_weight) = split_graph(data, edge_att, self.ratio)
                
                # Using confounded embeddings
                causal_x, causal_edge_index, causal_batch, _ = relabel(x, causal_edge_index, data.batch)
                conf_x, conf_edge_index, conf_batch, _ = relabel(x, conf_edge_index, data.batch)

                conf_node_weight = None
                causal_node_weight = None
            else:
                # NOT Using confounded embeddings for causal_x and conf_x
                (causal_x, causal_edge_index, causal_edge_attr, causal_batch, causal_node_weight), \
                    (conf_x, conf_edge_index, conf_edge_attr, conf_batch, conf_node_weight), \
                        (idx_keep, idx_remove) = split_graph_node(data, att, self.ratio, embed=x, use_input_feat=True)

                causal_edge_weight = lift_node_att_to_edge_att(causal_node_weight.unsqueeze(1), causal_edge_index)
                conf_edge_weight = lift_node_att_to_edge_att(conf_node_weight.unsqueeze(1), conf_edge_index)
        else:
            raise ValueError(f"{data.x.shape} {data.edge_index.shape}")
            causal_x, causal_edge_index, causal_edge_attr, causal_edge_weight, causal_batch = \
                x, data.edge_index, data.edge_attr, \
                float('inf') * torch.ones(data.edge_index.shape[1], device=data.x.device), \
                data.batch
            conf_x, conf_edge_index, conf_edge_attr, conf_edge_weight, conf_batch = None, None, None, None, None

        return (causal_x, causal_edge_index, causal_edge_attr, causal_node_weight, causal_edge_weight, causal_batch), \
               (conf_x, conf_edge_index, conf_edge_attr, conf_node_weight, conf_edge_weight, conf_batch), \
               (att_log_logits, att)

    
    def get_full_graph_explanation(self, *args, **kwargs):
        assert not self.learn_edge_att

        data = kwargs.get('data') or None

        x = self.gnn_node(*args, **kwargs) # extract node embeddigns

        # WARNING this are log_logits
        att_log_logits = self.extractor(x, data.edge_index, data.batch)
        att = att_log_logits.sigmoid() # Added by me

        if data.edge_index.shape[1] != 0:
            # NOT Using confounded embeddings for causal_x and conf_x
            (causal_x, causal_edge_index, causal_edge_attr, causal_batch, causal_node_weight), \
                (conf_x, conf_edge_index, conf_edge_attr, conf_batch, conf_node_weight), \
                    (idx_keep, idx_remove) = split_graph_node(data, att, self.ratio, embed=x, use_input_feat=True)

            causal_edge_weight = lift_node_att_to_edge_att(causal_node_weight.unsqueeze(1), causal_edge_index)
            conf_edge_weight = lift_node_att_to_edge_att(conf_node_weight.unsqueeze(1), conf_edge_index)
        else:
            raise ValueError(f"{data.x.shape} {data.edge_index.shape}")
        
        att[idx_remove] = -1.0 #att[idx_remove]

        return (causal_x, causal_edge_index, causal_edge_attr, causal_node_weight, causal_edge_weight, causal_batch), \
               (conf_x, conf_edge_index, conf_edge_attr, conf_node_weight, conf_edge_weight, conf_batch), \
               (att)


def split_graph(data, edge_score, ratio):
    r"""
        Adapted from https://github.com/wuyxin/dir-gnn.
    """
    has_edge_attr = hasattr(data, 'edge_attr') and getattr(data, 'edge_attr') is not None

    new_idx_reserve, new_idx_drop, _, _, _ = sparse_topk(edge_score, data.batch[data.edge_index[0]], ratio, descending=True)
    new_causal_edge_index = data.edge_index[:, new_idx_reserve]
    new_conf_edge_index = data.edge_index[:, new_idx_drop]

    new_causal_edge_weight = edge_score[new_idx_reserve]
    new_conf_edge_weight = - edge_score[new_idx_drop]

    if has_edge_attr:
        new_causal_edge_attr = data.edge_attr[new_idx_reserve]
        new_conf_edge_attr = data.edge_attr[new_idx_drop]
    else:
        new_causal_edge_attr = None
        new_conf_edge_attr = None

    return (new_causal_edge_index, new_causal_edge_attr, new_causal_edge_weight), \
           (new_conf_edge_index, new_conf_edge_attr, new_conf_edge_weight)

def split_graph_node(data, node_score, ratio, embed, use_input_feat):
    r"""
        Adapted from https://github.com/wuyxin/dir-gnn.
    """
    batch = data.batch
    if batch is None:
        batch = torch.zeros(data.x.shape[0], device=data.x.device, dtype=torch.long)    

    new_idx_reserve, new_idx_drop, _, _, _ = sparse_topk(node_score.view(-1), batch, ratio, descending=True)

    new_causal_edge_index, new_causal_edge_attr = subgraph(
        subset=new_idx_reserve,
        edge_index=data.edge_index,
        edge_attr=data.edge_attr,
        relabel_nodes=True, # set to True for debug_subgraph_plot
        return_edge_mask=False,
        num_nodes=data.x.shape[0]
    )
    new_conf_edge_index, new_conf_edge_attr = subgraph(
        subset=new_idx_drop,
        edge_index=data.edge_index,
        edge_attr=data.edge_attr,
        relabel_nodes=True, # set to True for debug_subgraph_plot
        return_edge_mask=False,
        num_nodes=data.x.shape[0]
    )
    
    if use_input_feat:
        causal_x = data.x[new_idx_reserve]
        conf_x = data.x[new_idx_drop]
    else:
        causal_x = embed[new_idx_reserve]
        conf_x = embed[new_idx_drop]

    causal_batch = batch[new_idx_reserve]
    conf_batch = batch[new_idx_drop]

    causal_node_weight = node_score[new_idx_reserve]
    conf_node_weight = -1 * node_score[new_idx_drop]

    # S1 = Data(x=data.x[new_idx_reserve], edge_index=new_causal_edge_index, ori_node_idx=new_idx_reserve)
    # S2 = Data(x=data.x[new_idx_drop],    edge_index=new_conf_edge_index, ori_node_idx=new_idx_drop)
    # debug_subgraph_plot(data, S1, S2)

    return (causal_x, new_causal_edge_index, new_causal_edge_attr, causal_batch, causal_node_weight), \
            (conf_x, new_conf_edge_index, new_conf_edge_attr, conf_batch, conf_node_weight), \
                (new_idx_reserve, new_idx_drop)

def debug_subgraph_plot(original_graph, subgraph1, subgraph2):
    # Convert PyG graphs to NetworkX
    G = to_networkx(original_graph, node_attrs=['x'], to_undirected=True)
    SG1 = to_networkx(subgraph1, node_attrs=['x', 'ori_node_idx'], to_undirected=True)
    SG2 = to_networkx(subgraph2, node_attrs=['x', 'ori_node_idx'], to_undirected=True)
    
    # Compute layout based on original graph
    pos = nx.spring_layout(G, seed=42)
    
    # Plot all three graphs in a row
    fig, axes = plt.subplots(1, 3, figsize=(18, 9))
    
    
    for j, g in enumerate([G, SG1, SG2]):
        node_attr = list(nx.get_node_attributes(g, "x").values())
        
        node_colors = []
        for i in range(len(node_attr)):
            if node_attr[i] == [1., 0., 0., 0.]:
                node_colors.append("red")
            elif node_attr[i] == [0., 1., 0., 0.]:
                node_colors.append("blue")
            elif node_attr[i] == [0., 0., 1., 0.]:
                node_colors.append("green")
            elif node_attr[i] == [0., 0., 0., 1.]:
                node_colors.append("violet")
            else:
                node_colors.append("orange")
    
        axes[j].set_title("Original Graph")

        if j == 0:
            pos_here = pos
        else:
            pos_here = {k: pos[idx] for k, idx in enumerate(list(nx.get_node_attributes(g, "ori_node_idx").values()))}

        nx.draw(g, pos_here, ax=axes[j], with_labels=True, edge_color='gray', node_size=100, alpha=0.5, node_color=node_colors)
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("/home/azzolin/sedignn/expl_shortcut/GOOD/kernel/pipelines/plots/debug_subgraph.png", format='png')
    plt.close()


def split_batch(g):
    r"""
    Adopted from https://github.com/wuyxin/dir-gnn.
    """
    split = degree(g.batch[g.edge_index[0]], dtype=torch.long).tolist()
    edge_indices = torch.split(g.edge_index, split, dim=1)
    num_nodes = degree(g.batch, dtype=torch.long)
    cum_nodes = torch.cat([g.batch.new_zeros(1), num_nodes.cumsum(dim=0)[:-1]])
    num_edges = torch.tensor([e.size(1) for e in edge_indices], dtype=torch.long).to(g.x.device)
    cum_edges = torch.cat([g.batch.new_zeros(1), num_edges.cumsum(dim=0)[:-1]])

    return edge_indices, num_nodes, cum_nodes, num_edges, cum_edges


def relabel(x, edge_index, batch, pos=None):
    r"""
    Adopted from https://github.com/wuyxin/dir-gnn.
    """
    num_nodes = x.size(0)
    sub_nodes = torch.unique(edge_index)
    x = x[sub_nodes]
    batch = batch[sub_nodes]
    row, col = edge_index
    # remapping the nodes in the explanatory subgraph to new ids.
    node_idx = row.new_full((num_nodes,), -1)
    node_idx[sub_nodes] = torch.arange(sub_nodes.size(0), device=row.device)
    edge_index = node_idx[edge_index]
    if pos is not None:
        pos = pos[sub_nodes]
    return x, edge_index, batch, pos


def sparse_sort(src: torch.Tensor, index: torch.Tensor, dim=0, descending=False, eps=1e-12):
    r'''
    Adopted from https://github.com/rusty1s/pytorch_scatter/issues/48.
    '''
    f_src = src.float()
    f_min, f_max = f_src.min(dim)[0], f_src.max(dim)[0]
    # norm = (f_src - f_min) / (f_max - f_min + eps) + (index.float()+1) * (-1) ** int(descending)
    norm = f_src + index.float() * (-1) ** int(descending) # WARNING: Only if using sigmoided attention scores
    perm = norm.argsort(dim=dim, descending=descending)
    return src[perm], perm


def sparse_topk(src: torch.Tensor, index: torch.Tensor, ratio: float, dim=0, descending=False, eps=1e-12):
    r"""
    Sparse topk calculation.
    """
    rank, perm = sparse_sort(src, index, dim, descending, eps)
    num_nodes = degree(index, dtype=torch.long)
    k = (ratio * num_nodes.to(float)).ceil().to(torch.long)
    start_indices = torch.cat([torch.zeros((1, ), device=src.device, dtype=torch.long), num_nodes.cumsum(0)])
    mask = [torch.arange(k[i], dtype=torch.long, device=src.device) + start_indices[i] for i in range(len(num_nodes))]
    mask = torch.cat(mask, dim=0)
    mask = torch.zeros_like(index, device=index.device).index_fill(0, mask, 1).bool()
    topk_perm = perm[mask]
    exc_perm = perm[~mask]

    return topk_perm, exc_perm, rank, perm, mask

class ExtractorMLP(nn.Module):

    def __init__(self, config: Union[CommonArgs, Munch]):
        super().__init__()
        hidden_size = config.model.dim_hidden
        self.learn_edge_att = config.ood.extra_param[0]
        dropout_p = config.model.dropout_rate

        if self.learn_edge_att:
            self.feature_extractor = MLP([hidden_size * 2, hidden_size * 4, hidden_size, 1], dropout=dropout_p)
        else:
            self.feature_extractor = MLP([hidden_size * 1, hidden_size * 2, hidden_size, 1], dropout=dropout_p)

    def forward(self, emb, edge_index, batch):
        if self.learn_edge_att:
            col, row = edge_index
            f1, f2 = emb[col], emb[row]
            f12 = torch.cat([f1, f2], dim=-1)
            att_log_logits = self.feature_extractor(f12, batch[col])
        else:
            att_log_logits = self.feature_extractor(emb, batch)
        return att_log_logits


class BatchSequential(nn.Sequential):
    def forward(self, inputs, batch):
        for module in self._modules.values():
            inputs = module(inputs)
        return inputs


class MLP(BatchSequential):
    def __init__(self, channels, dropout, bias=True):
        m = []
        for i in range(1, len(channels)):
            m.append(nn.Linear(channels[i - 1], channels[i], bias))

            if i < len(channels) - 1:
                m.append(BatchNorm(channels[i]))
                m.append(nn.ReLU())
                m.append(nn.Dropout(dropout))

        super(MLP, self).__init__(*m)


def set_masks(mask: Tensor, model: nn.Module, node_mask:Tensor=None):
    r"""
    Adopted from https://github.com/wuyxin/dir-gnn.
    """
    for module in model.modules():
        if isinstance(module, MessagePassing):
            if __pyg_version__ == "2.4.0":
                module._fixed_explain = True
            else:
                module.__explain__ = True
                module._explain = True

            module._apply_sigmoid = False    
            module._edge_mask = mask

            if model.att_net.extractor.learn_edge_att == False:
                module._node_mask = node_mask


def clear_masks(model: nn.Module):
    r"""
    Adopted from https://github.com/wuyxin/dir-gnn.
    """
    for module in model.modules():
        if isinstance(module, MessagePassing):
            if __pyg_version__ == "2.4.0":
                module._fixed_explain = False
            else:
                module.__explain__ = False
                module._explain = False
            
            module._edge_mask = None
            module._node_mask = None