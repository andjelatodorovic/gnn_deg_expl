r"""
Interpretable and Generalizable Graph Learning via Stochastic Attention Mechanism
with Arthur-Morgana Robustness Certification

Modified version of GSAT that fully replaces ExtractorMLP with
ArthurMorganaGNNExplainer for certified explanation robustness.
ExtractorMLP and the GSAT pretrain path have been removed entirely.
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
import copy
from GOOD.utils.train import lift_node_att_to_edge_att

from .gnn_merlin_arthur import ArthurMorganaGNNExplainer, ExplanationBounds


@register.model_register
class GSATGNNs_MODIFIED(GNNBasic):
    """
    GSAT with Arthur-Morgana as the sole explanation extractor.
    The legacy ExtractorMLP and GSAT pretrain path have been removed.
    """

    def __init__(self, config: Union[CommonArgs, Munch], entropy_reg: bool = False):
        super(GSATGNNs_MODIFIED, self).__init__(config)

        config = copy.deepcopy(config)
        fe_kwargs = {'mitigation_readout': config.mitigation_readout}

        self.gnn = FeatExtractor(config, **fe_kwargs)

        # Arthur-Morgana hyperparameters from config.ood.extra_param.
        K               = config.ood.extra_param[0] if len(config.ood.extra_param) > 0 else 10
        epsilon         = config.ood.extra_param[1] if len(config.ood.extra_param) > 1 else 0.05
        game_iterations = config.ood.extra_param[2] if len(config.ood.extra_param) > 2 else 5
        merlin_lr       = config.ood.extra_param[3] if len(config.ood.extra_param) > 3 else 0.01
        morgana_steps   = config.ood.extra_param[4] if len(config.ood.extra_param) > 4 else 20
        morgana_lr      = config.ood.extra_param[5] if len(config.ood.extra_param) > 5 else 0.01

        # Lazy initialization: explainer is built on the first forward pass
        # so we know the actual node/edge count of the incoming batch.
        self.explainer = None
        self.explainer_config = {
            'K':               K,
            'epsilon':         epsilon,
            'game_iterations': game_iterations,
            'merlin_lr':       merlin_lr,
            'morgana_steps':   morgana_steps,
            'morgana_lr':      morgana_lr,
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

        # Arthur-Morgana always produces node-level masks; edge_att is derived.
        self.learn_edge_att = True
        self.config = config
        self.edge_mask = None
        self.entropy_reg = entropy_reg
        self.last_certificate = None

        print("GSAT initialized — Arthur-Morgana is the sole explanation extractor")
        print(f"  K={K}, epsilon={epsilon}, game_iters={game_iterations}")

    # ------------------------------------------------------------------
    # Checkpoint helpers — exclude per-batch explainer state
    # ------------------------------------------------------------------

    def state_dict(self, *args, **kwargs):
        """Exclude the lazy explainer from saved checkpoints.

        explainer.merlin.alpha is a per-batch scratch parameter that is
        re-created on every forward pass, so persisting it is both wrong
        and causes load_state_dict failures when the explainer hasn't been
        initialized yet.
        """
        sd = super().state_dict(*args, **kwargs)
        keys_to_drop = [k for k in sd if k.startswith("explainer.")]
        for k in keys_to_drop:
            del sd[k]
        return sd

    def load_state_dict(self, state_dict, strict=True):
        """Drop any stale explainer keys from old checkpoints, then load."""
        filtered = {k: v for k, v in state_dict.items()
                    if not k.startswith("explainer.")}
        missing, unexpected = super().load_state_dict(filtered, strict=False)

        real_missing    = [k for k in missing    if not k.startswith("explainer.")]
        real_unexpected = [k for k in unexpected if not k.startswith("explainer.")]

        if strict and (real_missing or real_unexpected):
            raise RuntimeError(
                f"Error loading state_dict:\n"
                f"  Missing keys:    {real_missing}\n"
                f"  Unexpected keys: {real_unexpected}"
            )
        return torch.nn.modules.module._IncompatibleKeys(real_missing, real_unexpected)

    # ------------------------------------------------------------------
    # Lazy explainer initialization
    # ------------------------------------------------------------------

    def _initialize_explainer(self, num_nodes: int, num_edges: int, device):
        """Create (or re-create) the ArthurMorganaGNNExplainer.

        MerlinOptimizer stores alpha as a fixed-size nn.Parameter, so it must
        be re-initialized whenever the batch node/edge count changes.
        """
        current_size = getattr(self, "_explainer_size", None)
        if self.explainer is None or current_size != (num_nodes, num_edges):
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
            self._explainer_size = (num_nodes, num_edges)

    # ------------------------------------------------------------------
    # Arthur-Morgana helper callbacks
    # ------------------------------------------------------------------

    def _gnn_forward(self, embeddings: Tensor, edge_index: Tensor, batch: Tensor) -> Tensor:
        """Arthur-Morgana helper — delegates to _classifier_forward."""
        return self._classifier_forward(embeddings, edge_index, batch)

    def _classifier_forward(self, embeddings: Tensor, edge_index: Tensor, batch: Tensor) -> Tensor:
        """Pool masked node embeddings and classify.

        Operates on hidden representations (shape [num_nodes, hidden_dim]),
        NOT raw node features — self.gnn must not be re-run here.
        """
        if batch is None:
            batch = torch.zeros(
                embeddings.size(0), dtype=torch.long, device=embeddings.device
            )

        batch_size = int(batch.max().item()) + 1 if batch.numel() > 0 else 1

        graph_emb = torch.zeros(
            batch_size, embeddings.size(-1),
            device=embeddings.device, dtype=embeddings.dtype,
        )
        graph_emb.index_add_(0, batch, embeddings)

        return self.classifierS(graph_emb)

    def _normal_logits_forward(self, *args, **kwargs) -> Tensor:
        """Standard GNN + classifier forward (edge masks must be set beforehand)."""
        if self.gnn_clf:
            return self.classifierS(self.gnn_clf(*args, **kwargs))
        else:
            return self.classifierS(self.gnn(*args, **kwargs))

    # ------------------------------------------------------------------
    # Attention conversion utilities
    # ------------------------------------------------------------------

    def _edge_attention_to_node_attention(self, edge_att: Tensor, data) -> Tensor:
        """Aggregate edge-level attention [num_edges] → node-level [num_nodes, 1]."""
        if edge_att.dim() > 1:
            edge_att = edge_att.view(-1)

        num_nodes = data.x.shape[0]
        src = data.edge_index[0]

        node_sum   = torch.zeros(num_nodes, device=edge_att.device, dtype=edge_att.dtype)
        node_count = torch.zeros(num_nodes, device=edge_att.device, dtype=edge_att.dtype)

        node_sum.index_add_(0, src, edge_att)
        node_count.index_add_(0, src, torch.ones_like(edge_att))

        node_att = node_sum / node_count.clamp(min=1.0)
        return node_att.unsqueeze(-1)

    def _attention_to_edge_attention(self, att: Tensor, data) -> Tensor:
        """Return edge-level attention [num_edges].

        Accepts either node-level [num_nodes] / [num_nodes, 1]
        or edge-level [num_edges] / [num_edges, 1].
        """
        flat      = att.view(-1)
        num_nodes = data.x.shape[0]
        num_edges = data.edge_index.shape[1]

        if flat.numel() == num_edges:
            return flat

        if flat.numel() == num_nodes:
            edge_att = lift_node_att_to_edge_att(flat.unsqueeze(-1), data.edge_index)
            return edge_att.view(-1)

        raise RuntimeError(
            f"Attention size mismatch: got {flat.numel()}, "
            f"expected num_nodes={num_nodes} or num_edges={num_edges}"
        )

    def _dummy_attention(self, logits: Tensor, data) -> Tensor:
        """All-ones node attention used on the cheap no-grad path."""
        return torch.ones(data.x.shape[0], device=logits.device)

    def _soften_training_attention(self, att: Tensor, floor: float = 0.2) -> Tensor:
        """Soft residual version of Arthur-Morgana's hard mask.

        Maps 0 → floor, 1 → 1, so the classifier is never fully starved.
        """
        att = att.float()
        return floor + (1.0 - floor) * att

    def _attention_to_log_logits(self, att: Tensor) -> Tensor:
        """Convert attention probabilities to safe GSAT-style logits [N, 1]."""
        att = att.float()

        if att.dim() == 1:
            att = att.unsqueeze(-1)

        att = torch.nan_to_num(att, nan=0.5, posinf=0.999, neginf=0.001)

        eps = 0.05
        att = att.clamp(eps, 1.0 - eps)

        logits = torch.logit(att)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=6.0, neginf=-6.0)
        logits = logits.clamp(-3.0, 3.0)

        return logits

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, *args, **kwargs):
        data = kwargs.get("data")
        task = getattr(self.config, "task", None)

        # ---- Cheap path: no-grad evaluation (e.g. GOOD's eval_train) ----
        if (not torch.is_grad_enabled()) and task != "eval_metric":
            logits = self._normal_logits_forward(*args, **kwargs)
            att_log_logits = torch.zeros(
                data.x.shape[0], 1,
                device=logits.device, dtype=logits.dtype,
            )
            att = att_log_logits.sigmoid()
            self.last_certificate = None
            return logits, att_log_logits, att

        # ---- Cheap path: test inference ----
        if (not self.training) and task == "test":
            logits = self._normal_logits_forward(*args, **kwargs)
            att    = self._dummy_attention(logits, data)
            self.last_certificate = None
            att_log_logits = self._attention_to_log_logits(att)
            return logits, att_log_logits, att

        # ---- Full Arthur-Morgana path (train + eval_metric) ----
        if data is None:
            raise ValueError("Arthur-Morgana GSAT requires `data` in kwargs.")

        if (not hasattr(data, "batch")) or data.batch is None:
            data.batch = torch.zeros(
                data.x.shape[0], dtype=torch.long, device=data.x.device
            )

        # Get hidden node embeddings from the GNN backbone.
        emb = self.gnn(*args, without_readout=True, **kwargs)

        # (Re-)init explainer if batch shape changed.
        self._initialize_explainer(
            num_nodes=emb.shape[0],
            num_edges=data.edge_index.shape[1],
            device=emb.device,
        )

        # Run the Merlin-Morgana-Arthur game.
        # torch.enable_grad() lets eval_metric run Arthur-Morgana even when
        # GOOD wraps outer evaluation in torch.no_grad().
        with torch.enable_grad():
            explanation = self.explainer(
                embeddings=emb.detach(),
                edge_index=data.edge_index,
                batch=data.batch,
                target=data.y,
            )

        att = explanation.mask.float()
        if self.training and task == "train":
            att = self._soften_training_attention(att, floor=0.2)
        att = att.clamp(0.05, 0.95)

        self.last_certificate = {
            "L_min":               explanation.clean_loss,
            "L_max":               explanation.worst_loss,
            "robustness_margin":   explanation.robustness_margin,
            "convergence_history": explanation.convergence_history,
        }

        edge_att = self._attention_to_edge_attention(att, data)
        node_att = att if att.dim() == 2 else att.unsqueeze(-1)

        set_masks(edge_att, self, node_att)
        logits = self._normal_logits_forward(*args, **kwargs)
        clear_masks(self)

        self.edge_mask = edge_att

        att_log_logits = self._attention_to_log_logits(att)
        returned_att   = att_log_logits.sigmoid().clamp(0.05, 0.95)

        return logits, att_log_logits, returned_att

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    @staticmethod
    def concrete_sample(att_log_logit, temp, training):
        """Legacy stub — not called when Arthur-Morgana is active."""
        if training:
            random_noise = torch.empty_like(att_log_logit).uniform_(1e-10, 1 - 1e-10)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            att_bern = ((att_log_logit + random_noise) / temp).sigmoid()
        else:
            att_bern = att_log_logit.sigmoid()
        return att_bern

    @torch.no_grad()
    def probs(self, *args, **kwargs):
        out    = self(*args, **kwargs)
        logits = out[0] if isinstance(out, tuple) else out
        if logits.shape[-1] > 1:
            return logits.softmax(dim=-1)
        else:
            return logits.sigmoid()

    def get_subgraph(self, *args, **kwargs):
        """Return node-level explanation for GOOD faithfulness metrics.

        GOOD's generate_binary_explanations expects node_scores with shape
        [num_nodes, 1] because it later calls squeeze(1).
        """
        logits, att_log_logits, att = self.forward(*args, **kwargs)

        node_scores = att if att is not None else att_log_logits.sigmoid()

        if node_scores.dim() == 1:
            node_scores = node_scores.unsqueeze(-1)

        node_scores = torch.nan_to_num(
            node_scores.float(), nan=0.5, posinf=0.95, neginf=0.05,
        ).clamp(0.05, 0.95)

        return node_scores, node_scores, logits


# ---------------------------------------------------------------------------
# Mask helpers
# ---------------------------------------------------------------------------

def set_masks(mask: Tensor, model: nn.Module, node_mask: Tensor = None):
    """Set edge and node masks on all MessagePassing conv layers.

    Both _edge_mask (message passing) and _node_mask (weighted readout)
    are always set together to avoid the AIA fallback in Pooling.py.
    """
    if model.gnn_clf is None:
        modules = model.gnn.encoder.convs.modules()
    else:
        modules = model.gnn_clf.encoder.convs.modules()

    node_mask_for_pool = None
    if node_mask is not None:
        node_mask_for_pool = (
            node_mask.unsqueeze(-1) if node_mask.dim() == 1 else node_mask
        )

    for module in modules:
        if isinstance(module, MessagePassing):
            if __pyg_version__ == "2.4.0":
                module._fixed_explain = True
            else:
                module.__explain__ = True
                module._explain    = True

            module._apply_sigmoid = False
            module._edge_mask     = mask
            module._node_mask     = node_mask_for_pool


def clear_masks(model: nn.Module):
    """Clear edge/node masks from all MessagePassing conv layers."""
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
                module._explain    = False

            module._edge_mask = None
            module._node_mask = None


class ExtractorMLP(nn.Module):
    """Removed — kept as a tombstone to catch accidental imports."""

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "ExtractorMLP has been removed. "
            "Use ArthurMorganaGNNExplainer (via GSATGNNs_MODIFIED) instead."
        )

    def forward(self, *args, **kwargs):
        raise RuntimeError(
            "ExtractorMLP has been removed. "
            "Use ArthurMorganaGNNExplainer (via GSATGNNs_MODIFIED) instead."
        )