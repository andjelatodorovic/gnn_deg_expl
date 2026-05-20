r"""
Interpretable and Generalizable Graph Learning via Stochastic Attention Mechanism
with Arthur-Morgana Robustness Certification

Modified version of GSAT that replaces ExtractorMLP with ArthurMorganaGNNExplainer
for certified explanation robustness.
"""

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import BatchNorm
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import is_undirected, to_undirected, coalesce
from torch_sparse import transpose
from torch_geometric import __version__ as __pyg_version__

from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from .BaseGNN import GNNBasic
from .Classifiers import Classifier
from .GINs import FeatExtractor
from .GINvirtualnode import vFeatExtractor
import copy
from GOOD.utils.splitting import split_graph, relabel
from GOOD.utils.train import lift_node_att_to_edge_att

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
class GSAT(GNNBasic):
    """
    GSAT with Arthur-Morgana explanation extractor.
    
    Replaces gradient-based ExtractorMLP with game-theoretic framework for
    certified robustness guarantees on node selection explanations.
    """

    def __init__(self, config: Union[CommonArgs, Munch], entropy_reg: bool=False):
        super(GSAT, self).__init__(config)
        
        config = copy.deepcopy(config)
        fe_kwargs = {'mitigation_readout': config.mitigation_readout}

        self.gnn = FeatExtractor(config, **fe_kwargs)
        
        # ============================================================
        # REPLACE: self.extractor = ExtractorMLP(config)
        # WITH: Arthur-Morgana Explainer
        # ============================================================
        
        # Parse Arthur-Morgana hyperparameters from config
        K = config.ood.extra_param[0] if len(config.ood.extra_param) > 0 else 10
        epsilon = config.ood.extra_param[1] if len(config.ood.extra_param) > 1 else 0.05
        game_iterations = config.ood.extra_param[2] if len(config.ood.extra_param) > 2 else 5
        merlin_lr = config.ood.extra_param[3] if len(config.ood.extra_param) > 3 else 0.01
        morgana_steps = config.ood.extra_param[4] if len(config.ood.extra_param) > 4 else 20
        morgana_lr = config.ood.extra_param[5] if len(config.ood.extra_param) > 5 else 0.01
        
        # Lazy initialization (will be set on first forward pass)
        self.explainer = None
        self.explainer_config = {
            'K': K,
            'epsilon': epsilon,
            'game_iterations': game_iterations,
            'merlin_lr': merlin_lr,
            'morgana_steps': morgana_steps,
            'morgana_lr': morgana_lr,
        }

        if config.mitigation_sampling == "raw":
            print("Init CLASSIFIER")
            fe_kwargs["gnn_clf_layer"] = config.model.gnn_clf_layer
            fe_kwargs["no_bias"] = True
            self.gnn_clf = FeatExtractor(config, **fe_kwargs)
            print(f"Using mitigation_sampling==raw with {config.model.gnn_clf_layer} layers")
        else:
            self.gnn_clf = None

        self.classifierS = Classifier(config, is_linear=False)
        
        self.learn_edge_att = config.ood.extra_param[0] if len(config.ood.extra_param) > 0 else True
        self.config = config
        self.edge_mask = None
        self.entropy_reg = entropy_reg
        self.last_certificate = None  # Store robustness bounds
        
        print(f"GSAT initialized with Arthur-Morgana explainer")
        print(f"  K={self.explainer_config['K']}, epsilon={self.explainer_config['epsilon']}, "
              f"game_iters={self.explainer_config['game_iterations']}")

    def _initialize_explainer(self, num_nodes: int, num_edges: int, device):
        """Lazy initialization of Arthur-Morgana explainer."""
        if self.explainer is None:
            print(f"Initializing ArthurMorganaGNNExplainer: {num_nodes} nodes, {num_edges} edges")
            self.explainer = ArthurMorganaGNNExplainer(
                num_nodes=num_nodes,
                num_edges=num_edges,
                gnn_forward_fn=self._gnn_forward,
                classifier_fn=self._classifier_forward,
                K=self.explainer_config['K'],
                epsilon=self.explainer_config['epsilon'],
                game_iterations=self.explainer_config['game_iterations'],
                merlin_lr=self.explainer_config['merlin_lr'],
                morgana_steps=self.explainer_config['morgana_steps'],
                morgana_lr=self.explainer_config['morgana_lr'],
                device=device,
            ).to(device)

    def _gnn_forward(self, embeddings: Tensor, edge_index: Tensor, batch: Tensor) -> Tensor:
        """Forward through GNN feature extractor."""
        # This is a simplified version - calls the gnn
        return self.gnn.forward(embeddings=embeddings, edge_index=edge_index, batch=batch, 
                               without_readout=False)

    def _classifier_forward(self, embeddings: Tensor, edge_index: Tensor, batch: Tensor) -> Tensor:
        """Forward through GNN + classifier."""
        gnn_out = self.gnn.forward(embeddings=embeddings, edge_index=edge_index, batch=batch,
                                   without_readout=False)
        logits = self.classifierS(gnn_out)
        return logits

    def forward(self, *args, **kwargs):
        r"""
        The GSAT model with Arthur-Morgana explanation extraction.

        Args:
            *args (list): argument list for the use of arguments_read.
            **kwargs (dict): key word arguments for the use of arguments_read.

        Returns (Tensor):
            Label predictions and robustness certificate for loss calculations.
        """
        data = kwargs.get('data')
        
        # ============================================================
        # Standard GNN embedding
        # ============================================================
        emb = self.gnn(*args, without_readout=True, **kwargs)
        
        # ============================================================
        # ARTHUR-MORGANA: Explanation Extraction with Certification
        # ============================================================
        
        # Lazy init explainer on first forward pass
        self._initialize_explainer(emb.shape[0], data.edge_index.shape[1], emb.device)
        
        # Run game-theoretic explanation extraction
        if self.training:
            # Full game during training
            explanation = self.explainer(
                embeddings=emb.detach(),  # Detach to avoid gradient issues
                edge_index=data.edge_index,
                batch=data.batch,
                target=data.y,
            )
        else:
            # Single iteration during evaluation (faster)
            explainer_eval = ArthurMorganaGNNExplainer(
                num_nodes=emb.shape[0],
                num_edges=data.edge_index.shape[1],
                gnn_forward_fn=self._gnn_forward,
                classifier_fn=self._classifier_forward,
                K=self.explainer_config['K'],
                epsilon=self.explainer_config['epsilon'],
                game_iterations=1,  # Single iteration for speed
                merlin_lr=self.explainer_config['merlin_lr'],
                morgana_steps=self.explainer_config['morgana_steps'],
                morgana_lr=self.explainer_config['morgana_lr'],
                device=emb.device,
            ).to(emb.device)
            explanation = explainer_eval(
                embeddings=emb.detach(),
                edge_index=data.edge_index,
                batch=data.batch,
                target=data.y,
            )
        
        # Extract mask from explanation
        att = explanation.mask  # Binary mask (num_nodes,)
        
        # Store certificate for logging/analysis
        self.last_certificate = {
            'L_min': explanation.clean_loss,
            'L_max': explanation.worst_loss,
            'robustness_margin': explanation.robustness_margin,
            'convergence_history': explanation.convergence_history,
        }
        
        # ============================================================
        # Apply mask to embeddings and forward through classifier
        # ============================================================
        
        if self.learn_edge_att:
            # Lift node mask to edge mask (if needed for compatibility)
            if is_undirected(data.edge_index):
                if self.config.average_edge_attn == "default":
                    nodesize = data.x.shape[0]
                    # Simple approach: duplicate mask for each edge
                    edge_att = att[data.edge_index[0]]
                else:
                    # Use node mask directly
                    edge_att = att
            else:
                edge_att = att
        else:
            edge_att = lift_node_att_to_edge_att(att, data.edge_index)
        
        # Apply masks
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

        # ============================================================
        # Return: logits, None (no att_log_logits in game framework), att mask
        # ============================================================
        return logits, None, att

    def sampling(self, att_log_logits, training, mitigation_expl_scores):
        """Legacy method - not used with Arthur-Morgana."""
        # Kept for compatibility, but not called anymore
        att = self.concrete_sample(att_log_logits, temp=1, training=training)
        return att

    @staticmethod
    def concrete_sample(att_log_logit, temp, training):
        """Legacy method - not used with Arthur-Morgana."""
        if training:
            random_noise = torch.empty_like(att_log_logit).uniform_(1e-10, 1 - 1e-10)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            att_bern = ((att_log_logit + random_noise) / temp).sigmoid()
        else:
            att_bern = (att_log_logit).sigmoid()
        return att_bern
    
    @torch.no_grad()
    def probs(self, *args, **kwargs):        
        out = self(*args, **kwargs)
        logits = out[0] if isinstance(out, tuple) else out
        return logits.softmax(dim=-1)


# ============================================================================
# Utility functions (copied from original GSAT for masking)
# ============================================================================

def set_masks(mask, model, node_att=None):
    r"""Set the masks in the model."""
    for module in model.modules():
        if isinstance(module, MessagePassing):
            if isinstance(mask, Tensor):
                module.__explains__ = True
                module.__edge_mask__ = mask
                if node_att is not None:
                    module.__node_att__ = node_att
            else:
                module.__explains__ = False


def clear_masks(model):
    r"""Clear the masks in the model."""
    for module in model.modules():
        if isinstance(module, MessagePassing):
            module.__explains__ = False
            if hasattr(module, '__edge_mask__'):
                del module.__edge_mask__
            if hasattr(module, '__node_att__'):
                del module.__node_att__


# ============================================================================
# Optional: ExtractorMLP stub for backward compatibility
# ============================================================================

class ExtractorMLP(nn.Module):
    """Deprecated - kept for backward compatibility only."""
    
    def __init__(self, config):
        super().__init__()
        print("WARNING: ExtractorMLP is deprecated. GSAT now uses ArthurMorganaGNNExplainer.")
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError("ExtractorMLP has been replaced by ArthurMorganaGNNExplainer")
