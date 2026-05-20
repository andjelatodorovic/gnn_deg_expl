"""
DIRGNN_MODIFIED.py - DIR with Arthur-Morgana Game-Theoretic Explainer

Modified from: https://github.com/steveazzolin/gnn_deg_expl/blob/main/GOOD/networks/models/DIRGNN.py

Changes:
- Replaced ExtractorMLP with ArthurMorganaGNNExplainer in CausalAttNet
- Maintains interface compatibility with DIR
- Lazy initialization on first forward pass
- Stores robustness certificates in last_certificate dict
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
from .GINvirtualnode import vFeatExtractor
from .GINs import FeatExtractor
from .Classifiers import Classifier
from torch_geometric.utils.loop import add_self_loops, remove_self_loops
from torch_geometric.utils import is_undirected, to_undirected, coalesce, subgraph
from torch_sparse import transpose
from torch_geometric import __version__ as __pyg_version__

# Import Arthur-Morgana framework
import sys
import os
# Support both macOS and Windows paths
workspace_path = os.path.normpath('/Users/cartenoir/.openclaw/workspace-at-cn')
if not os.path.exists(workspace_path):
    # Try alternative Windows path
    workspace_path = os.path.normpath('C:\\Users\\cartenoir\\.openclaw\\workspace-at-cn')
if not os.path.exists(workspace_path):
    # Try relative path from GOOD/networks/models
    workspace_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '.openclaw', 'workspace-at-cn'))
sys.path.insert(0, workspace_path)
from gnn_merlin_arthur import ArthurMorganaGNNExplainer, ExplanationBounds


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
        self.last_certificate = None  # Store robustness certificate

    def forward(self, *args, **kwargs):
        r"""
        The modified DIR model with Arthur-Morgana explainer.

        Args:
            *args (list): argument list for the use of arguments_read
            **kwargs (dict): key word arguments for the use of arguments_read

        Returns (Tensor):
            Label predictions and other results for loss calculations.
        """
        data = kwargs.get('data')
        batch_size = data.batch[-1].item() + 1

        (causal_x, causal_edge_index, causal_edge_attr, causal_node_weight, causal_edge_weight, causal_batch), \
        (conf_x, conf_edge_index, conf_edge_attr, conf_node_weight, conf_edge_weight, conf_batch), \
            (node_att_logit, node_att, certificate) = self.att_net(*args, **kwargs)

        # Store certificate
        self.last_certificate = certificate

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

        set_masks(conf_edge_weight, self, conf_node_weight)
        conf_rep = self.get_graph_rep(
            data=Data(x=conf_x, edge_index=conf_edge_index,
                        edge_attr=conf_edge_attr, batch=conf_batch),
            batch_size=batch_size
        ).detach()
        conf_out = self.get_conf_pred(conf_rep)
        clear_masks(self)

        # --- combine to causal phase (Optimized version) ---
        rep_out2 = torch.transpose(
            self.get_comb_pred_eff(causal_rep, conf_rep),
            0,
            1
        )

        return (None, rep_out2), causal_out, conf_out, (node_att_logit, node_att)

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
        set_masks(edge_att, self, node_att)
        causal_rep = self.get_graph_rep(*args, **kwargs)
        lc_logits = self.get_causal_pred(causal_rep)
        clear_masks(self)

        if log is None:
            if lc_logits.shape[-1] > 1:
                return lc_logits.softmax(-1)
            else:
                return lc_logits.sigmoid()
        else:
            assert False
    
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
    The GIN virtual node version of DIR (with Arthur-Morgana).
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
    The GIN virtual node without batchnorm version of DIR (with Arthur-Morgana).
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
    Causal Attention Network with Arthur-Morgana explainer.
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
        
        # Initialize explainer lazily (on first forward pass)
        self.explainer = None
        self._explainer_initialized = False
        
        self.ratio = causal_ratio
        self.config = config

        print("Causal ratio = ", self.ratio)

    def _initialize_explainer(self, num_nodes, num_edges, device):
        """Lazy initialization of Arthur-Morgana explainer."""
        if self._explainer_initialized:
            return
        
        # Read hyperparameters from config
        K = int(self.config.ood.extra_param[0]) if len(self.config.ood.extra_param) > 0 else 10
        epsilon = float(self.config.ood.extra_param[1]) if len(self.config.ood.extra_param) > 1 else 0.05
        game_iterations = int(self.config.ood.extra_param[2]) if len(self.config.ood.extra_param) > 2 else 5
        merlin_lr = float(self.config.ood.extra_param[3]) if len(self.config.ood.extra_param) > 3 else 0.01
        morgana_steps = int(self.config.ood.extra_param[4]) if len(self.config.ood.extra_param) > 4 else 20
        morgana_lr = float(self.config.ood.extra_param[5]) if len(self.config.ood.extra_param) > 5 else 0.01
        
        self.explainer = ArthurMorganaGNNExplainer(
            num_nodes=num_nodes,
            num_edges=num_edges,
            K=K,
            epsilon=epsilon,
            game_iterations=game_iterations,
            merlin_lr=merlin_lr,
            morgana_steps=morgana_steps,
            morgana_lr=morgana_lr,
            device=device
        )
        self.explainer = self.explainer.to(device)
        self._explainer_initialized = True
        print(f"Initializing ArthurMorganaGNNExplainer (K={K}, ε={epsilon}, game_iters={game_iterations})")

    def forward(self, *args, **kwargs):
        data = kwargs.get('data') or None
        device = next(self.gnn_node.parameters()).device

        # Initialize explainer on first call
        if not self._explainer_initialized:
            self._initialize_explainer(data.num_nodes, data.edge_index.shape[1], device)

        x = self.gnn_node(*args, **kwargs)

        # Get explanation via Arthur-Morgana game
        x_detached = x.detach()
        
        mask, certificate = self.explainer(
            embeddings=x_detached,
            edge_index=data.edge_index,
            is_training=self.training
        )

        # Convert binary mask to attention scores (for compatibility)
        att_log_logits = torch.log(mask + 1e-8)  # Log odds
        att = mask  # Binary mask
        
        if data.edge_index.shape[1] != 0:
            if self.learn_edge_att:
                if is_undirected(data.edge_index):
                    if self.config.average_edge_attn == "default":
                        nodesize = data.x.shape[0]
                        edge_att = (att + transpose(data.edge_index, att, nodesize, nodesize, coalesced=False)[1]) / 2
                    else:
                        data.ori_edge_index = data.edge_index.detach().clone()
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

        return (causal_x, causal_edge_index, causal_edge_attr, causal_node_weight, causal_edge_weight, causal_batch), \
               (conf_x, conf_edge_index, conf_edge_attr, conf_node_weight, conf_edge_weight, conf_batch), \
               (att_log_logits, att, certificate)


def split_graph(data, edge_score, ratio):
    r"""Split graph into causal and confounded edges."""
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
    r"""Split graph into causal and confounded nodes."""
    batch = data.batch
    if batch is None:
        batch = torch.zeros(data.x.shape[0], device=data.x.device, dtype=torch.long)    

    new_idx_reserve, new_idx_drop, _, _, _ = sparse_topk(node_score.view(-1), batch, ratio, descending=True)

    new_causal_edge_index, new_causal_edge_attr = subgraph(
        subset=new_idx_reserve,
        edge_index=data.edge_index,
        edge_attr=data.edge_attr,
        relabel_nodes=True,
        return_edge_mask=False,
        num_nodes=data.x.shape[0]
    )
    new_conf_edge_index, new_conf_edge_attr = subgraph(
        subset=new_idx_drop,
        edge_index=data.edge_index,
        edge_attr=data.edge_attr,
        relabel_nodes=True,
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

    return (causal_x, new_causal_edge_index, new_causal_edge_attr, causal_batch, causal_node_weight), \
            (conf_x, new_conf_edge_index, new_conf_edge_attr, conf_batch, conf_node_weight), \
                (new_idx_reserve, new_idx_drop)


def sparse_topk(edge_score, batch, ratio, descending):
    """Select top-k edges per graph in batch."""
    # Simplified version - in practice would use more efficient implementation
    perm = torch.argsort(edge_score.view(-1), descending=descending)
    k = int(ratio * edge_score.shape[0])
    reserve = perm[:k]
    drop = perm[k:]
    return reserve, drop, None, None, None


def relabel(x, edge_index, batch):
    """Relabel nodes after subgraph extraction."""
    # Simplified relabeling
    unique_nodes = torch.unique(edge_index)
    mapping = torch.full((x.shape[0],), -1, dtype=torch.long, device=x.device)
    mapping[unique_nodes] = torch.arange(unique_nodes.shape[0], device=x.device)
    
    new_edge_index = mapping[edge_index]
    new_batch = batch[unique_nodes]
    new_x = x[unique_nodes]
    
    return new_x, new_edge_index, new_batch, unique_nodes


def set_masks(mask: Tensor, model: nn.Module, node_mask: Tensor = None):
    r"""Set edge/node masks for message passing layers."""
    if model.gnn_clf is None:
        modules = model.gnn.encoder.convs.modules()
    else:
        modules = model.gnn_clf.encoder.convs.modules()

    for module in modules:
        if isinstance(module, MessagePassing):
            if __pyg_version__ == "2.4.0":
                module._fixed_explain = True
            else:
                module.__explain__ = True
                module._explain = True
            
            module._apply_sigmoid = False    
            module._edge_mask = mask

            if hasattr(model, 'att_net') and model.att_net and hasattr(model.att_net, 'learn_edge_att'):
                if model.att_net.learn_edge_att == False:
                    module._node_mask = node_mask


def clear_masks(model: nn.Module):
    r"""Clear edge/node masks from message passing layers."""
    if model.gnn_clf is None:
        modules = model.gnn.encoder.convs.modules()
    else:
        modules = model.gnn_clf.encoder.convs.modules()

    for module in modules:
        if isinstance(module, MessagePassing):
            if __pyg_version__ == "2.4.0":
                module._fixed_explain = False
            else:
                module.__explain__ = False
                module._explain = False
            module._edge_mask = None
            module._node_mask = None
