r"""
Interpretable and Generalizable Graph Learning via Stochastic Attention Mechanism
with the Merlin-Morgana-Arthur Framework.

Based on: Wäldchen et al. (2022) "Interpretability Guarantees with
Merlin-Arthur Classifiers" — https://arxiv.org/abs/2206.00759

Protocol
--------
  Merlin  (honest prover)       — selects the subgraph that best supports the
                                  correct label.  Uses cooperative gradient
                                  descent on a soft node mask.

  Morgana (adversarial prover)  — selects the subgraph that most confuses
                                  Arthur, i.e. maximises the classification
                                  loss via PGD attack on node embeddings.

  Arthur  (classifier)          — at training time sees both Merlin's faithful
                                  subgraph and Morgana's adversarial subgraph,
                                  learning to be correct on the first and robust
                                  against the second.

                                  At INFERENCE time sees only Merlin's subgraph.

What changed from the original GSATGNNs_MODIFIED.py
----------------------------------------------------
  - ArthurMorganaGNNExplainer (black-box game engine) is replaced by two
    explicit extractors: MerlinExtractor and MorganaExtractor, both living
    inside this file as first-class nn.Module members of the GSAT model.
  - forward() now returns FIVE values at training time:
      logits_merlin, att_log_logits, att_merlin, logits_morgana, att_morgana
    and THREE at inference (logits_merlin, att_log_logits, att_merlin).
  - The combined loss is computed by merlin_morgana_arthur_loss() below.
  - Checkpoint handling, mask helpers, and all utility methods are preserved
    unchanged from the original file.
"""

import copy
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

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
from GOOD.utils.train import lift_node_att_to_edge_att


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _graph_loss(logits: Tensor, target: Tensor) -> Tensor:
    
    if logits.shape[-1] == 1:
        return F.binary_cross_entropy_with_logits(
            logits.view(-1),
            target.float().view(-1),
        )

    return F.cross_entropy(
        logits,
        target.long().view(-1),
    )



# MerlinExtractor — 


class MerlinExtractor(nn.Module):
    """
    Merlin selects the K nodes whose embeddings best support the correct label.

    Internally runs Adam for `merlin_steps` steps on a soft mask alpha, then
    thresholds to a hard binary top-K mask per graph.

    Objective:  minimise  CE( Arthur(emb * sigmoid(alpha)), y )
                        + lambda * ||sigmoid(alpha)||_1    (sparsity)

    The alpha parameter is re-created each forward call to match the current
    batch size, so no state leaks between batches.
    """

    def __init__(
        self,
        merlin_steps: int = 100,
        merlin_lr:    float = 0.001,
        sparsity_lambda: float = 0.1,
        K: int = 10,
        min_k: int = 1,
        max_k: Optional[int] = None,
    ):
        super().__init__()
        self.merlin_steps = merlin_steps
        self.merlin_lr = merlin_lr
        self.sparsity_lambda = sparsity_lambda
        self.K = float(K)
        self.min_k = int(min_k)
        self.max_k = None if max_k is None else int(max_k)

    def forward(
        self,
        embeddings:   Tensor,   
        edge_index:   Tensor,   
        batch:        Tensor,   
        classifier_fn,          
        target:       Tensor,   
    ) -> tuple[Tensor, Tensor]:
        """
        Returns
        -------
        soft_mask   : sigmoid(alpha) in (0,1)^N  — used as soft attention
        binary_mask : hard top-K binary mask in {0,1}^N
        """
        N = embeddings.shape[0]
        alpha = torch.zeros(N, device=embeddings.device, requires_grad=True)
        optimizer = torch.optim.Adam([alpha], lr=self.merlin_lr)

        for _ in range(self.merlin_steps):
            optimizer.zero_grad()
            m_soft = torch.sigmoid(alpha)
            masked_emb = embeddings * m_soft.unsqueeze(1)
            logits = classifier_fn(masked_emb, edge_index, batch)
            loss = _graph_loss(logits, target) + self.sparsity_lambda * m_soft.mean()
           

            loss.backward()
            optimizer.step()

        # Hard top-K mask per graph
        soft_mask = torch.sigmoid(alpha).detach()
        binary_mask = torch.zeros_like(soft_mask)
        K_value = float(self.K)

        for graph_id in batch.unique(sorted=True):
            idx = (batch == graph_id).nonzero(as_tuple=True)[0]
            n_graph = idx.numel()

            if 0.0 < K_value < 1.0:
                k_graph = int(round(K_value * n_graph))
                k_graph = max(self.min_k, k_graph)

                if self.max_k is not None:
                    k_graph = min(k_graph, self.max_k)
            else:
                k_graph = int(round(K_value))

            k_graph = max(1, min(k_graph, n_graph))

            _, local_top = torch.topk(soft_mask[idx], k_graph)
            binary_mask[idx[local_top]] = 1.0

        return soft_mask, binary_mask


# ---------------------------------------------------------------------------
# MorganaExtractor — adversarial prover
# ---------------------------------------------------------------------------

class MorganaExtractor(nn.Module):
    """
    Morgana finds the embedding perturbation that MAXIMISES Arthur's loss,
    i.e. the subgraph that most confuses the classifier.

    Uses Projected Gradient Descent (PGD) within an ell-inf ball of radius
    epsilon around the original embeddings:

        delta* = argmax_{||delta||_inf <= epsilon}  CE( Arthur(emb + delta), y )

    Multiple random restarts are used to avoid local maxima.

    At training time, Arthur is also presented with these adversarially
    perturbed embeddings and must learn to classify them correctly (soundness).
    At inference time, MorganaExtractor is NOT called.
    """

    def __init__(
        self,
        epsilon:         float = 0.05,
        morgana_steps:   int   = 20,
        morgana_lr:      float = 0.01,
        morgana_restarts: int  = 3,
    ):
        super().__init__()
        self.epsilon          = epsilon
        self.morgana_steps    = morgana_steps
        self.morgana_lr       = morgana_lr
        self.morgana_restarts = morgana_restarts

    def forward(
        self,
        embeddings:   Tensor,   # [N, D]  original (detached) node embeddings
        edge_index:   Tensor,
        batch:        Tensor,
        classifier_fn,          # callable: (emb, edge_index, batch) -> logits
        target:       Tensor,
    ) -> tuple[Tensor, float]:
        """
        Returns
        -------
        adv_embeddings : embeddings + best delta  [N, D]
        worst_loss     : scalar float — CE under best attack
        """
        best_loss  = -float("inf")
        best_delta = None

        for _ in range(self.morgana_restarts):
            delta = (torch.rand_like(embeddings) * 2 - 1) * self.epsilon
            delta = delta.detach().requires_grad_(True)

            for _ in range(self.morgana_steps):
                logits = classifier_fn(embeddings + delta, edge_index, batch)
                loss   = _graph_loss(logits, target)
                loss.backward()
                with torch.no_grad():
                    # Gradient ASCENT: move delta to increase loss
                    delta = delta + self.morgana_lr * torch.sign(delta.grad)
                    delta = delta.clamp(-self.epsilon, self.epsilon)
                delta = delta.detach().requires_grad_(True)

            with torch.no_grad():
                final_loss = _graph_loss(
                    classifier_fn(embeddings + delta, edge_index, batch), target
                ).item()

            if final_loss > best_loss:
                best_loss  = final_loss
                best_delta = delta.detach().clone()

        if best_delta is None:
            best_delta = torch.zeros_like(embeddings)

        return embeddings + best_delta, best_loss


# ---------------------------------------------------------------------------
# Main model — GSATGNNs_MODIFIED
# ---------------------------------------------------------------------------

@register.model_register
class GSATGNNs_MODIFIED(GNNBasic):
    """
    GSAT with the Merlin-Morgana-Arthur framework.

    Merlin and Morgana are explicit nn.Module members of this class.
    Arthur is self.classifierS (unchanged from original GSAT).

    Training
    --------
    forward() returns five values:
        logits_merlin    — Arthur on Merlin's faithful subgraph
        att_log_logits   — Merlin's soft mask as GSAT-style logits  [N, 1]
        att_merlin       — Merlin's soft attention weights           [N, 1]
        logits_morgana   — Arthur on Morgana's adversarial subgraph
        att_morgana      — ones mask (Morgana perturbs embeddings, not topology)

    Pass all five to merlin_morgana_arthur_loss() to compute the combined loss.

    Inference
    ---------
    forward() returns three values (logits_morgana and att_morgana are None):
        logits_merlin, att_log_logits, att_merlin

    Configuration  (config.ood.extra_param)
    ----------------------------------------
    [0]  K                number of nodes Merlin selects per graph   (default 10)
    [1]  epsilon          Morgana's ell-inf perturbation budget       (default 0.05)
    [2]  game_iterations  Merlin-Morgana rounds per batch             (default 1)
    [3]  merlin_lr        Adam lr for Merlin's mask                   (default 0.01)
    [4]  morgana_steps    PGD steps per Morgana restart               (default 20)
    [5]  morgana_lr       PGD step size                               (default 0.01)
    [6]  morgana_weight   lambda weight on soundness loss term        (default 1.0)
    """

    def __init__(self, config: Union[CommonArgs, Munch], entropy_reg: bool = False):
        super(GSATGNNs_MODIFIED, self).__init__(config)

        config = copy.deepcopy(config)
        fe_kwargs = {"mitigation_readout": config.mitigation_readout}

        # Shared GNN backbone — produces node embeddings for both provers.
        self.gnn = FeatExtractor(config, **fe_kwargs)

        # Read hyperparameters from config
        ep = config.ood.extra_param
        K                = ep[0] if len(ep) > 0 else 0.3
        epsilon          = ep[1] if len(ep) > 1 else 0.05
        game_iterations  = ep[2] if len(ep) > 2 else 1
        merlin_lr        = ep[3] if len(ep) > 3 else 0.01
        morgana_steps    = ep[4] if len(ep) > 4 else 20
        morgana_lr       = ep[5] if len(ep) > 5 else 0.01
        self.morgana_weight = float(ep[6]) if len(ep) > 6 else 1.0
        self.game_iterations = int(game_iterations)



        print(f"[MMA raw extra_param] {config.ood.extra_param}")

        print(
            f"[MMA parsed params] "
            f"K={K}, epsilon={epsilon}, "
            f"game_iterations={self.game_iterations}, "
            f"merlin_lr={merlin_lr}, "
            f"morgana_steps={morgana_steps}, "
            f"morgana_lr={morgana_lr}, "
            f"morgana_weight={self.morgana_weight}"
        )

        assert self.game_iterations >= 1, f"game_iterations must be >=1, got {self.game_iterations}"
        assert morgana_steps >= 1, f"morgana_steps must be >=1, got {morgana_steps}"
        assert float(epsilon) > 0, f"epsilon must be >0, got {epsilon}"
        assert float(self.morgana_weight) >= 0, f"morgana_weight must be >=0, got {self.morgana_weight}"

        # --- Merlin: honest prover ---
        self.merlin = MerlinExtractor(
            K=K,
            merlin_lr=merlin_lr,
            merlin_steps=100,
            sparsity_lambda=0.05,
        )

        # --- Morgana: adversarial prover ---
        self.morgana = MorganaExtractor(
            epsilon=epsilon,
            morgana_steps=morgana_steps,
            morgana_lr=morgana_lr,
            morgana_restarts=3,
        )

        # Optional separate classifier GNN (raw mode, unchanged from original)
        if config.mitigation_sampling == "raw":
            print("[MMA] Init CLASSIFIER GNN (raw mode)")
            fe_kwargs["gnn_clf_layer"] = config.model.gnn_clf_layer
            fe_kwargs["no_bias"] = True
            self.gnn_clf = FeatExtractor(config, **fe_kwargs)
        else:
            self.gnn_clf = None

        # --- Arthur: the verifier/classifier ---
        self.classifierS = Classifier(config, is_linear=False)

        self.learn_edge_att = True
        self.config         = config
        self.edge_mask      = None
        self.entropy_reg    = entropy_reg
        self.last_certificate = None
        self.logits_morgana = None
        self.att_morgana = None
        self.morgana_loss = None
        self.morgana_loss_value = None

        print(
            f"[MMA-GSAT] Merlin-Morgana-Arthur initialised. "
            f"K={K}  epsilon={epsilon}  game_iterations={self.game_iterations}  "
            f"morgana_weight={self.morgana_weight}"
        )

   

    def state_dict(self, *args, **kwargs):
        """Merlin/Morgana have no persistent per-batch state, so nothing to
        strip here.  Method kept for API compatibility."""
        return super().state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=strict)

    # ------------------------------------------------------------------
    # Internal classifier callback used by Merlin and Morgana
    # ------------------------------------------------------------------

    def _pool_and_classify(
        self,
        embeddings: Tensor,
        edge_index: Tensor,
        batch:      Tensor,
    ) -> Tensor:
        """
        Sum-pool masked node embeddings over each graph, then classify.

        This operates on hidden representations [N, D], NOT raw features.
        self.gnn is NOT re-run here — embeddings are already the backbone output.
        """
        if batch is None:
            batch = torch.zeros(
                embeddings.size(0), dtype=torch.long, device=embeddings.device
            )
        batch_size = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
        graph_emb  = torch.zeros(
            batch_size, embeddings.size(-1),
            device=embeddings.device, dtype=embeddings.dtype,
        )
        graph_emb.index_add_(0, batch, embeddings)
        return self.classifierS(graph_emb)

    def _normal_logits_forward(self, *args, **kwargs) -> Tensor:
        """Standard GNN + classifier forward (masks must be set beforehand)."""
        if self.gnn_clf:
            return self.classifierS(self.gnn_clf(*args, **kwargs))
        return self.classifierS(self.gnn(*args, **kwargs))

    


    def _attention_to_edge_attention(self, att: Tensor, data) -> Tensor:
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
        return torch.ones(data.x.shape[0], device=logits.device)

    def _soften_training_attention(self, att: Tensor, floor: float = 0.15) -> Tensor:
        att = att.float()
        return floor + (1.0 - floor) * att

    def _attention_to_log_logits(self, att: Tensor) -> Tensor:
        att = att.float()
        if att.dim() == 1:
            att = att.unsqueeze(-1)
        att    = torch.nan_to_num(att, nan=0.5, posinf=0.999, neginf=0.001)
        att    = att.clamp(0.05, 0.95)
        logits = torch.logit(att)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=6.0, neginf=-6.0)
        return logits.clamp(-3.0, 3.0)

    # ------------------------------------------------------------------
    # Forward

    def forward(self, *args, **kwargs):
        """
        Training  → returns (logits_merlin, att_log_logits, att_merlin,
                              logits_morgana, att_morgana)
        Inference → returns (logits_merlin, att_log_logits, att_merlin,
                              None, None)
        """
        data = kwargs.get("data")
        task = getattr(self.config, "task", None)

        #fix attempt
        if (not torch.is_grad_enabled()) and task not in ("eval_metric", "test"):

            logits         = self._normal_logits_forward(*args, **kwargs)
            att_log_logits = torch.zeros(data.x.shape[0], 1,
                                         device=logits.device, dtype=logits.dtype)
            att            = att_log_logits.sigmoid()
            self.last_certificate = None
            self.logits_morgana = None
            self.att_morgana = None
            self.morgana_loss = None
            self.morgana_loss_value = None
            return logits, att_log_logits, att

        # Cheap path: test inference 
        
        if (not self.training) and task == "test":
            logits         = self._normal_logits_forward(*args, **kwargs)
            att            = self._dummy_attention(logits, data)
            att_log_logits = self._attention_to_log_logits(att)
            self.last_certificate = None
            self.logits_morgana   = None
            self.att_morgana      = None
            self.morgana_loss     = None
            self.morgana_loss_value = None
            return logits, att_log_logits, att

    
    # other than in cases when when task == "eval_metric" (get_subgraph needs real Merlin scores), zero att scores
        if (not torch.is_grad_enabled()) and task not in ("eval_metric",):
            logits         = self._normal_logits_forward(*args, **kwargs)
            att_log_logits = torch.zeros(data.x.shape[0], 1,
                                        device=logits.device, dtype=logits.dtype)
            att            = att_log_logits.sigmoid()
            self.last_certificate = None
            self.logits_morgana   = None
            self.att_morgana      = None
            self.morgana_loss     = None
            self.morgana_loss_value = None
            return logits, att_log_logits, att

        # Ensure batch assignment exists
        if data is None:
            raise ValueError("GSATGNNs_MODIFIED requires `data` in kwargs.")
        if not hasattr(data, "batch") or data.batch is None:
            data.batch = torch.zeros(data.x.shape[0], dtype=torch.long,
                                     device=data.x.device)

        # ------------------------------------------------------------------
        # 1. Shared GNN backbone → node embeddings (no readout yet)
        # ------------------------------------------------------------------
        emb = self.gnn(*args, without_readout=True, **kwargs)

        # Detach so that Merlin's and Morgana's inner optimisation loops
        # do not corrupt the backbone's gradients.
        emb_detached = emb.detach()

        # ------------------------------------------------------------------
        # 2. Merlin: honest prover
        #    Runs game_iterations rounds; uses the last mask produced.
        # ------------------------------------------------------------------
        with torch.enable_grad():
            for _ in range(self.game_iterations):
                soft_mask, binary_mask = self.merlin(
                    embeddings=emb_detached,
                    edge_index=data.edge_index,
                    batch=data.batch,
                    classifier_fn=self._pool_and_classify,
                    target=data.y,
                )

        # Soften binary mask at training time so Arthur is never fully starved.
        att_merlin = binary_mask.float()
        if self.training:
            att_merlin = self._soften_training_attention(att_merlin)
        att_merlin = att_merlin.clamp(0.05, 0.95)

        # Apply Merlin's mask and run Arthur
        edge_att_merlin = self._attention_to_edge_attention(att_merlin, data)
        node_att_merlin = att_merlin.unsqueeze(-1)
        set_masks(edge_att_merlin, self, node_att_merlin)
        logits_merlin = self._normal_logits_forward(*args, **kwargs)
        clear_masks(self)

        self.edge_mask = edge_att_merlin

        att_log_logits = self._attention_to_log_logits(att_merlin)
        returned_att   = att_log_logits.sigmoid().clamp(0.05, 0.95)

        # Morgana: adversarial prover — training only
        #    Finds the perturbation of embeddings that maximises Arthur's loss.
        #    Arthur must then learn to classify correctly even on this input.

        if self.training:
            with torch.enable_grad():
                adv_emb, worst_loss = self.morgana(
                    embeddings=emb_detached,
                    edge_index=data.edge_index,
                    batch=data.batch,
                    classifier_fn=self._pool_and_classify,
                    target=data.y,
                )

            self.last_certificate = {
                "L_min":             _graph_loss(logits_merlin, data.y).item(),
                "L_max":             worst_loss,
                "robustness_margin": worst_loss - _graph_loss(logits_merlin, data.y).item(),
            }

            # Run Arthur on Morgana's adversarial embeddings.
            logits_morgana = self._pool_and_classify(
                adv_emb, data.edge_index, data.batch
            )
            att_morgana = torch.ones_like(att_merlin)  # Morgana uses all nodes

        else:
            # Inference: Arthur only sees Merlin's subgraph.
            logits_morgana = None
            att_morgana    = None
            self.last_certificate = None

        # GOOD/GSAT expects exactly three outputs:
        #   logits, attention_logits, attention
        

        # Here my Morgana loss is stored internally 
        self.logits_morgana = logits_morgana
        self.att_morgana = att_morgana

        # Optional Morgana  loss.
        # GOOD still receives exactly three outputs, but the training loop can read self.morgana_loss and add it to the normal loss.
        if self.training and logits_morgana is not None:
            self.morgana_loss = _graph_loss(logits_morgana, data.y)
            self.morgana_loss_value = float(self.morgana_loss.detach().cpu())
        else:
            self.morgana_loss = None
            self.morgana_loss_value = None

        return logits_merlin, att_log_logits, returned_att

   

    @staticmethod
    def concrete_sample(att_log_logit, temp, training):
        """Legacy stub — kept for API compatibility."""
        if training:
            noise = torch.empty_like(att_log_logit).uniform_(1e-10, 1 - 1e-10)
            noise = torch.log(noise) - torch.log(1.0 - noise)
            return ((att_log_logit + noise) / temp).sigmoid()
        return att_log_logit.sigmoid()

    @torch.no_grad()
    def probs(self, *args, **kwargs):
        out    = self(*args, **kwargs)
        logits = out[0]
        return logits.softmax(dim=-1) if logits.shape[-1] > 1 else logits.sigmoid()
    
    @torch.no_grad()
    def predict_from_subgraph(
        self,
        edge_att=False,
        log=None,
        eval_kl=None,
        node_att=False,
        edge_attn=None,
        *args,
        **kwargs,
    ):
        """GSAT-compatible prediction from an externally supplied subgraph.

        This mirrors the original GSAT implementation:
        - set masks
        - run the normal classifier path
        - clear masks
        - return probabilities, not raw logits

        The only addition is support for GOOD's `edge_attn` keyword and
        conversion from node_att to edge_att when only node_att is supplied.
        """

        data = kwargs.get("data", None)

        # GOOD sometimes passes edge_attn instead of edge_att.
        if edge_attn is not None:
            edge_att = edge_attn

        # If only node_att is provided, derive edge attention from node attention.
        # This is needed for BAColorGVIsolated, where explanations are node-level.
        if (
            data is not None
            and node_att is not False
            and node_att is not None
            and (edge_att is False or edge_att is None)
        ):
            if node_att.dim() == 1:
                node_att = node_att.unsqueeze(-1)

            node_att = torch.nan_to_num(
                node_att.float(),
                nan=0.5,
                posinf=1.0,
                neginf=0.0,
            ).clamp(0.0, 1.0)

            edge_att = self._attention_to_edge_attention(node_att, data)

        set_masks(edge_att, self, node_att)

        if self.gnn_clf:
            lc_logits = self.classifierS(self.gnn_clf(*args, **kwargs))
        else:
            lc_logits = self.classifierS(self.gnn(*args, **kwargs))

        clear_masks(self)

        # IMPORTANT: counter_fid expects probabilities/log-probabilities,
        # not raw logits.
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
                    new_logits = torch.zeros(
                        (lc_logits.shape[0], lc_logits.shape[1] + 1),
                        device=lc_logits.device,
                    )
                    new_logits[:, 1] = new_logits[:, 1] + lc_logits.squeeze(1)
                    new_logits[:, 0] = 1 - new_logits[:, 1]
                    new_logits[new_logits == 0.0] = 1e-10
                    return new_logits.log()
                else:
                    return lc_logits.sigmoid().log()



    def get_subgraph(self, *args, **kwargs):
        """Return node-level explanation for GOOD faithfulness metrics."""
        original_task = getattr(self.config, "task", None)
        self.config.task = "eval_metric"
        try:
            logits, att_log_logits, att = self.forward(*args, **kwargs)
        finally:
            self.config.task = original_task  # always restore, even if forward() raises

        node_scores = att if att is not None else att_log_logits.sigmoid()
        if node_scores.dim() == 1:
            node_scores = node_scores.unsqueeze(-1)
        node_scores = torch.nan_to_num(
            node_scores.float(), nan=0.5, posinf=0.95, neginf=0.05
        ).clamp(0.05, 0.95)

        return node_scores, node_scores, logits

    


def merlin_morgana_arthur_loss(
    logits_merlin:  Tensor,
    logits_morgana: Optional[Tensor],
    att_log_logits: Tensor,
    target:         Tensor,
    morgana_weight: float = 1.0,
    info_loss_coef: float = 1.0,
) -> tuple[Tensor, dict]:
    """
    Combined Merlin-Morgana-Arthur training loss.


    Parameters
    ----------
    logits_merlin   : Arthur's output on Merlin's subgraph          [B, C]
    logits_morgana  : Arthur's output on Morgana's adv. embeddings  [B, C] or None
    att_log_logits  : Merlin's soft mask as log-logits              [N, 1]
    target          : ground-truth labels                            [B]
    morgana_weight  : lambda — weight on soundness term
    info_loss_coef  : mu — weight on GSAT info-bottleneck term

    Returns
    -------
    total_loss : scalar Tensor
    loss_dict  : dict of individual components for logging
    """
    # Completeness: Arthur classifies Merlin's faithful subgraph correctly.
    l_completeness = _graph_loss(logits_merlin, target)

    # Soundness: Arthur must not be fooled by Morgana's adversarial subgraph // goal is that Arthur minimizes this
    l_soundness = (
        _graph_loss(logits_morgana, target)
        if logits_morgana is not None
        else torch.tensor(0.0, device=logits_merlin.device)
    )

    # GSAT information-bottleneck regularisation on Merlin's soft mask.
    # Encourages Merlin to select a sparse, informative subgraph.
    att = att_log_logits.sigmoid()
    prior = torch.tensor(0.7, device=att.device)
    l_info = (
        att * torch.log(att / prior + 1e-8)
        + (1 - att) * torch.log((1 - att) / (1 - prior) + 1e-8)
    ).mean()

    total = (
        l_completeness
        + morgana_weight * l_soundness
        + info_loss_coef * l_info
    )

    return total, {
        "loss_completeness": l_completeness.item(),
        "loss_soundness":    l_soundness.item(),
        "loss_info":         l_info.item(),
        "loss_total":        total.item(),
    }


#some helper functions around masks - unchanged from GSAT

def set_masks(mask: Tensor, model: nn.Module, node_mask: Tensor = None):
    """Set edge and node masks on all MessagePassing conv layers."""
    modules = (
        model.gnn.encoder.convs.modules()
        if model.gnn_clf is None
        else model.gnn_clf.encoder.convs.modules()
    )
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
    modules = (
        model.gnn.encoder.convs.modules()
        if model.gnn_clf is None
        else model.gnn_clf.encoder.convs.modules()
    )
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

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "ExtractorMLP has been removed. "
            "GSATGNNs_MODIFIED now uses MerlinExtractor and MorganaExtractor."
        )

    def forward(self, *args, **kwargs):
        raise RuntimeError(
            "ExtractorMLP has been removed. "
            "GSATGNNs_MODIFIED now uses MerlinExtractor and MorganaExtractor."
        )