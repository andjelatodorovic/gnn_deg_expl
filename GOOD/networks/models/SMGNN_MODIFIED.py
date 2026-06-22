"""
SMGNN_MODIFIED.py - SMGNN with Arthur-Morgana Game-Theoretic Explainer

Modified from: https://github.com/steveazzolin/gnn_deg_expl/blob/main/GOOD/networks/models/SMGNN.py

Changes:
- Replaced ExtractorMLP with ArthurMorganaGNNExplainer
- Maintains interface compatibility with SMGNN
"""

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import InstanceNorm, BatchNorm
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import is_undirected, to_undirected, degree, coalesce
from torch_sparse import transpose
from torch_geometric import __version__ as __pyg_version__
from torch_scatter import scatter_softmax

from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from .BaseGNN import GNNBasic
from .Classifiers import Classifier
from .GINs import FeatExtractor
from .GINvirtualnode import vFeatExtractor
import copy
from GOOD.utils.splitting import split_graph, relabel
from GOOD.utils.train import lift_node_att_to_edge_att

import sys
import os

from gnn_merlin_arthur import ArthurMorganaGNNExplainer, ExplanationBounds

@register.model_register
class SMGNN(GNNBasic):

    def __init__(self, config: Union[CommonArgs, Munch]):
        super(SMGNN, self).__init__(config)
        
        config = copy.deepcopy(config)
        fe_kwargs = {'mitigation_readout': config.mitigation_readout}

        self.gnn = FeatExtractor(config, **fe_kwargs)
        
        # Initialize explainer lazily (on first forward pass)
        self.explainer = None
        self._explainer_initialized = False

        if config.mitigation_sampling == "raw":
            print("Init CLASSIFIER")
            fe_kwargs["gnn_clf_layer"] = config.model.gnn_clf_layer
            fe_kwargs["no_bias"] = True
            self.gnn_clf = FeatExtractor(config, **fe_kwargs)
            print(f"Using mitigation_sampling==raw with {config.model.gnn_clf_layer} layers")
        else:
            self.gnn_clf = None

        self.classifierS = Classifier(config, is_linear=False)

        self.learn_edge_att = config.ood.extra_param[0]
        self.config = config
        self.edge_mask = None
        self.last_certificate = None  # Store robustness certificate
        print("Using mitigation_expl_scores:", config.mitigation_expl_scores)
        
    def _initialize_explainer(self, num_nodes, num_edges, device):
        """Lazy initialization of Arthur-Morgana explainer."""
        if self._explainer_initialized:
            return
        
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
        r"""
        The modified SMGNN with Arthur-Morgana explainer.

        Args:
            *args (list): argument list for the use of arguments_read
            **kwargs (dict): key word arguments for the use of arguments_read

        Returns (Tensor):
            Label predictions and other results for loss calculations.
        """
        data = kwargs.get('data')
        device = next(self.parameters()).device
        
        # Initialize explainer on first call
        if not self._explainer_initialized:
            self._initialize_explainer(data.num_nodes, data.edge_index.shape[1], device)
        
        # GNN embedding
        emb = self.gnn(*args, without_readout=True, **kwargs)
        
        # Get explanation via Arthur-Morgana game
        emb_detached = emb.detach()
        
        # Run game-theoretic explanation
        mask, certificate = self.explainer(
            embeddings=emb_detached,
            edge_index=data.edge_index,
            is_training=self.training
        )
        
        self.last_certificate = certificate
        
        # Convert binary mask to attention scores (for compatibility)
        if self.learn_edge_att:
            att_log_logits = torch.log(mask + 1e-8)  # Log odds for compatibility
            att = mask  # Binary mask
        else:
            # Node-level: aggregate edge mask to node mask
            from torch_scatter import scatter_add
            col, row = data.edge_index
            node_mask = scatter_add(mask.squeeze(-1), col, dim=0, dim_size=data.num_nodes)
            node_mask = node_mask / (degree(col, num_nodes=data.num_nodes, dtype=torch.float).clamp(min=1) + 1e-8)
            att_log_logits = torch.log(node_mask.unsqueeze(-1) + 1e-8)
            att = node_mask.unsqueeze(-1)

        if self.learn_edge_att:
            if is_undirected(data.edge_index):
                if self.config.average_edge_attn == "default":
                    nodesize = data.x.shape[0]
                    edge_att = (att + transpose(data.edge_index, att, nodesize, nodesize, coalesced=False)[1]) / 2
                else:
                    data.ori_edge_index = data.edge_index.detach().clone()
                    if not data.edge_attr is None:
                        edge_index_sorted, edge_attr_sorted = coalesce(data.ori_edge_index, data.edge_attr, is_sorted=False)
                        data.edge_attr = edge_attr_sorted
                    if hasattr(data, "edge_gt") and not data.edge_gt is None:
                        edge_index_sorted, edge_gt_sorted = coalesce(data.ori_edge_index, data.edge_gt, is_sorted=False)
                        data.edge_gt = edge_gt_sorted
                    data.edge_index, edge_att = to_undirected(data.edge_index, att.squeeze(-1), reduce="mean")
            else:
                edge_att = att
        else:
            edge_att = lift_node_att_to_edge_att(att, data.edge_index)

        set_masks(edge_att, self, att)
        
        if self.gnn_clf:
            logits = self.classifierS(self.gnn_clf(*args, **kwargs))
        else:
            if kwargs.get('pretrain'):
                logits = self.classifierS(self.gnn(*args, **kwargs).detach())
            else:
                logits = self.classifierS(self.gnn(*args, **kwargs))

        clear_masks(self)
        self.edge_mask = edge_att
        return logits, att_log_logits, att

    def sampling(self, att_log_logits, training, mitigation_expl_scores, batch=None):
        """Stub for backward compatibility."""
        return torch.sigmoid(att_log_logits)

    @staticmethod
    def concrete_sample(att_log_logit, temp, training, batch):
        """Stub for backward compatibility."""
        if training:
            random_noise = torch.empty_like(att_log_logit).uniform_(1e-10, 1 - 1e-10)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            att_bern = ((att_log_logit + random_noise) / temp).sigmoid()
        else:
            att_bern = torch.clamp(
                torch.sigmoid(att_log_logit), 
                min=0.00001,
                max=0.99999
            )
        return att_bern
    
    @torch.no_grad()
    def probs(self, *args, **kwargs):
        out = self(*args, **kwargs)
        
        if len(out) == 5:
            logits, att, edge_att, _, _ = out
        else:
            logits, att, edge_att = out

        if logits.shape[-1] > 1:
            return logits.softmax(dim=1)
        else:
            return logits.sigmoid()
    
    @torch.no_grad()
    def log_probs(self, eval_kl=False, *args, **kwargs):
        out = self(*args, **kwargs)

        if len(out) == 5:
            logits, att, edge_att, _, _ = out
        else:
            logits, att, edge_att = out
            
        if logits.shape[-1] > 1:
            return logits.log_softmax(dim=1)
        else:
            if eval_kl:
                logits = logits.sigmoid()
                new_logits = torch.zeros((logits.shape[0], logits.shape[1]+1), device=logits.device)
                new_logits[:, 1] = new_logits[:, 1] + logits.squeeze(1)
                new_logits[:, 0] = 1 - new_logits[:, 1]
                new_logits[new_logits == 0.] = 1e-10
                return new_logits.log()
            else:
                return logits.sigmoid().log()
        
    @torch.no_grad()
    def predict_from_subgraph(self, edge_att=False, log=None, eval_kl=None, node_att=False, *args, **kwargs):
        set_masks(edge_att, self, node_att)

        if self.gnn_clf:
            lc_logits = self.classifierS(self.gnn_clf(*args, **kwargs))
        else:
            lc_logits = self.classifierS(self.gnn(*args, **kwargs))
        
        clear_masks(self)

        if log is None:
            if lc_logits.shape[-1] > 1:
                return lc_logits.softmax(dim=1)
            else:
                return lc_logits.sigmoid()
        else:
            assert not (eval_kl is None)
            if lc_logits.shape[-1] > 1:
                return lc_logits.log_softmax(dim=1)
            else:
                if eval_kl:
                    lc_logits = lc_logits.sigmoid()
                    new_logits = torch.zeros((lc_logits.shape[0], lc_logits.shape[1]+1), device=lc_logits.device)
                    new_logits[:, 1] = new_logits[:, 1] + lc_logits.squeeze(1)
                    new_logits[:, 0] = 1 - new_logits[:, 1]
                    new_logits[new_logits == 0.] = 1e-10
                    return new_logits.log()
                else:
                    return lc_logits.sigmoid().log()
    
    def get_subgraph(self, *args, **kwargs):
        logits, att_log_logits, att = self.forward(*args, **kwargs)
        return self.edge_mask, att, logits

@register.model_register
class SMGNNv(SMGNN):
    r"""
    The GIN virtual node version of SMGNN (with Arthur-Morgana).
    """

    def __init__(self, config: Union[CommonArgs, Munch]):
        super(SMGNNv, self).__init__(config)
        exit("virtual nodes not in use")
        fe_kwargs = {'mitigation_readout': config.mitigation_readout}
        self.gnn = vFeatExtractor(config, **fe_kwargs)

        if config.mitigation_sampling == "raw":
            config.model.model_layer = 1
            self.gnn_clf = vFeatExtractor(config)
        else:
            self.gnn_clf = None


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

            if hasattr(model, 'explainer') and model.explainer and hasattr(model.explainer, 'learn_edge_att'):
                if model.explainer.learn_edge_att == False:
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
