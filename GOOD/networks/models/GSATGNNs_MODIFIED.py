import copy
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.data import Data
from torch_geometric.utils import subgraph
from torch_geometric import __version__ as __pyg_version__

from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from .BaseGNN import GNNBasic
from .Classifiers import Classifier
from .GINs import FeatExtractor
from GOOD.utils.train import lift_node_att_to_edge_att


def _graph_loss(logits: Tensor, target: Tensor) -> Tensor:
    if logits.shape[-1] == 1:
        return F.binary_cross_entropy_with_logits(logits.view(-1), target.float().view(-1))
    return F.cross_entropy(logits, target.long().view(-1))


def _consistency_loss(logits: Tensor, target_probs: Tensor) -> Tensor:
    """
    Divergence between predictions on a (masked / intervened) graph and the
    clean full-graph prediction. This is the quantity suff_cause measures
    (|p_clean - p_perturbed| on the predicted class), so we optimize it directly.
    target_probs must be detached.
    """
    if logits.shape[-1] == 1:
        return F.binary_cross_entropy_with_logits(logits.view(-1), target_probs.view(-1))
    return F.kl_div(logits.log_softmax(dim=-1), target_probs, reduction="batchmean")


class MerlinExtractor(nn.Module):
    """
    Honest prover. Gradient descent on a per-batch soft node mask to find
    the subgraph that best supports the correct label.

    classifier_fn(node_mask) -> logits must be the full masked-GNN path.
    """

    def __init__(self, K=0.3, merlin_lr=0.005, merlin_steps=20,
                 sparsity_lambda=0.001, min_k=1, max_k=None, consistency_weight=0.5):
        super().__init__()
        self.K = float(K)
        self.merlin_lr = merlin_lr
        self.merlin_steps = merlin_steps
        self.sparsity_lambda = sparsity_lambda
        self.min_k = int(min_k)
        self.max_k = None if max_k is None else int(max_k)
        # weight on matching the clean full-graph prediction (sufficiency objective)
        self.consistency_weight = float(consistency_weight)

    def _topk(self, soft_mask, batch):
        binary_mask = torch.zeros_like(soft_mask)
        for gid in batch.unique(sorted=True):
            idx = (batch == gid).nonzero(as_tuple=True)[0]
            n = idx.numel()
            k = int(round(self.K * n)) if 0.0 < self.K < 1.0 else int(round(self.K))
            k = max(self.min_k, k)
            if self.max_k is not None:
                k = min(k, self.max_k)
            k = max(1, min(k, n))
            _, top = torch.topk(soft_mask[idx], k)
            binary_mask[idx[top]] = 1.0
        return binary_mask

    def forward(self, embeddings, edge_index, batch, classifier_fn, target,
                target_probs=None):
        N = embeddings.shape[0]
        alpha = torch.zeros(N, device=embeddings.device, requires_grad=True)
        opt = torch.optim.Adam([alpha], lr=self.merlin_lr)

        for _ in range(self.merlin_steps):
            opt.zero_grad()
            m = torch.sigmoid(alpha)
            logits = classifier_fn(m)
            loss = _graph_loss(logits, target) + self.sparsity_lambda * m.mean()
            
            if target_probs is not None and self.consistency_weight > 0:
                loss = loss + self.consistency_weight * _consistency_loss(logits, target_probs)
            loss.backward()
            opt.step()

        soft_mask = torch.sigmoid(alpha).detach()
        return soft_mask, self._topk(soft_mask, batch)


class MorganaExtractor(nn.Module):
    """
    Adversarial prover. PGD attack on node embeddings to maximise Arthur's
    classification loss. No learnable parameters — discarded after each batch.
    """

    def __init__(self, epsilon=0.05, morgana_steps=5, morgana_lr=0.01, morgana_restarts=3):
        super().__init__()
        self.epsilon = epsilon
        self.morgana_steps = morgana_steps
        self.morgana_lr = morgana_lr
        self.morgana_restarts = morgana_restarts

    def forward(self, embeddings, edge_index, batch, classifier_fn, target):
        best_loss, best_delta = -float("inf"), None

        for _ in range(self.morgana_restarts):
            delta = (torch.rand_like(embeddings) * 2 - 1) * self.epsilon
            delta = delta.detach().requires_grad_(True)

            for _ in range(self.morgana_steps):
                loss = _graph_loss(classifier_fn(embeddings + delta, edge_index, batch), target)
                loss.backward()
                with torch.no_grad():
                    delta = delta + self.morgana_lr * torch.sign(delta.grad)
                    delta = delta.clamp(-self.epsilon, self.epsilon)
                delta = delta.detach().requires_grad_(True)

            with torch.no_grad():
                final_loss = _graph_loss(
                    classifier_fn(embeddings + delta, edge_index, batch), target
                ).item()

            if final_loss > best_loss:
                best_loss = final_loss
                best_delta = delta.detach().clone()

        return embeddings + (best_delta if best_delta is not None else torch.zeros_like(embeddings)), best_loss


@register.model_register
class GSATGNNs_MODIFIED(GNNBasic):
    """
    GSAT + Merlin-Morgana-Arthur.

    """

    def __init__(self, config: Union[CommonArgs, Munch], entropy_reg: bool = False):
        super().__init__(config)
        config = copy.deepcopy(config)
        fe_kwargs = {"mitigation_readout": config.mitigation_readout}

        self.gnn = FeatExtractor(config, **fe_kwargs)

        ep = config.ood.extra_param or []
        K               = ep[0] if len(ep) > 0 else 0.3
        epsilon         = ep[1] if len(ep) > 1 else 0.05
        game_iterations = ep[2] if len(ep) > 2 else 1
        merlin_lr       = ep[3] if len(ep) > 3 else 0.005
        morgana_steps   = ep[4] if len(ep) > 4 else 5
        morgana_lr      = ep[5] if len(ep) > 5 else 0.01
        self.morgana_weight  = float(ep[6]) if len(ep) > 6 else 0.3
        self.suff_weight     = float(ep[7]) if len(ep) > 7 else 1.0
        self.clean_weight    = float(ep[8]) if len(ep) > 8 else 0.5
        consistency_weight   = float(ep[9]) if len(ep) > 9 else 0.5

        # number of suff_cause-style intervention samples per training batch
        self.n_suff_aug = 2

        # Hardcoded warmup; not controlled by extra_param.
        self.warmup_epochs = 30

        self.game_iterations = int(game_iterations)

        self.merlin  = MerlinExtractor(K=K, merlin_lr=merlin_lr, merlin_steps=20,
                                       sparsity_lambda=0.001,
                                       consistency_weight=consistency_weight)
        self.morgana = MorganaExtractor(epsilon=epsilon, morgana_steps=morgana_steps, morgana_lr=morgana_lr)

        if config.mitigation_sampling == "raw":
            fe_kwargs["gnn_clf_layer"] = config.model.gnn_clf_layer
            fe_kwargs["no_bias"] = True
            self.gnn_clf = FeatExtractor(config, **fe_kwargs)
        else:
            self.gnn_clf = None

        self.classifierS   = Classifier(config, is_linear=False)

        hidden_dim = (
            getattr(config.model, "dim_hidden", None)
            or getattr(config.model, "hidden_dim", None)
            or getattr(config.model, "hidden_size", None)
        )
        self.verifier_head = nn.Linear(int(hidden_dim), 1) if hidden_dim is not None \
                             else nn.LazyLinear(1)

        self.learn_edge_att = True
        self.config         = config
        self.edge_mask      = None
        self.entropy_reg    = entropy_reg
        self._clear_stored()

        print(f"[MMA] K={K} eps={epsilon} morgana_steps={morgana_steps} "
              f"morgana_lr={morgana_lr} morgana_weight={self.morgana_weight} "
              f"suff_weight={self.suff_weight} clean_weight={self.clean_weight} "
              f"consistency_weight={consistency_weight} n_suff_aug={self.n_suff_aug} "
              f"warmup_epochs={self.warmup_epochs}")

    def _clear_stored(self):
        self.last_certificate = None
        self.logits_morgana   = None
        self.att_morgana      = None
        self.morgana_loss     = None
        self.morgana_loss_value = None
        self.verifier_merlin  = None
        self.verifier_morgana = None
        self.suff_loss        = None
        self.clean_logits     = None

    def _pool(self, embeddings, batch):
        if batch is None:
            batch = torch.zeros(embeddings.size(0), dtype=torch.long, device=embeddings.device)
        B = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
        out = torch.zeros(B, embeddings.size(-1), device=embeddings.device, dtype=embeddings.dtype)
        out.index_add_(0, batch, embeddings)
        return out

    def _mean_pool(self, embeddings, batch):
        if batch is None:
            batch = torch.zeros(embeddings.size(0), dtype=torch.long, device=embeddings.device)
        B = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
        out = torch.zeros(B, embeddings.size(-1), device=embeddings.device, dtype=embeddings.dtype)
        out.index_add_(0, batch, embeddings)
        counts = torch.zeros(B, 1, device=embeddings.device, dtype=embeddings.dtype)
        ones = torch.ones(embeddings.size(0), 1, device=embeddings.device, dtype=embeddings.dtype)
        counts.index_add_(0, batch, ones)
        out = out / counts.clamp_min(1.0)
        return F.layer_norm(out, out.shape[-1:])

    def _pool_and_classify(self, embeddings, edge_index, batch):
        return self.classifierS(self._pool(embeddings, batch))

    def _run_gnn(self, *args, **kwargs):
        if self.gnn_clf:
            return self.classifierS(self.gnn_clf(*args, **kwargs))
        return self.classifierS(self.gnn(*args, **kwargs))

    def _node_to_edge_att(self, att, data):
        flat = att.view(-1)
        if flat.numel() == data.edge_index.shape[1]:
            return flat
        if flat.numel() == data.x.shape[0]:
            return lift_node_att_to_edge_att(flat.unsqueeze(-1), data.edge_index).view(-1)
        raise RuntimeError(
            f"Attention size mismatch: {flat.numel()} vs "
            f"nodes={data.x.shape[0]} edges={data.edge_index.shape[1]}"
        )

    def _to_log_logits(self, att):
        att = att.float()
        if att.dim() == 1:
            att = att.unsqueeze(-1)
        att = torch.nan_to_num(att, nan=0.5, posinf=0.999, neginf=0.001).clamp(0.05, 0.95)
        return torch.nan_to_num(torch.logit(att), nan=0.0, posinf=6.0, neginf=-6.0).clamp(-3.0, 3.0)

    def _in_warmup(self):
        current_epoch = getattr(self.config.train, "epoch", None)
        if current_epoch is None:
            return False
        return int(current_epoch) < self.warmup_epochs

    @staticmethod
    def _suff_augment(data, protect, node_keep_p=0.5, edge_drop_p=0.5):
        """
        Sample one suff_cause-style intervened version of the batch.
        Mirrors xai_utils.suff_cause + robust_fidelity(rfidm, p=0.5):
          - keep protected nodes (Merlin's R) + each other node w.p. node_keep_p,
            hard-delete the rest (relabeled subgraph, exactly like eval),
          - drop remaining edges w.p. edge_drop_p symmetrically, except edges
            inside R's induced subgraph, which are always kept.
        Returns (intervened data shim, batch_size).
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        N = x.size(0)
        device = x.device
        B = int(batch.max().item()) + 1 if batch.numel() > 0 else 1

        keep = (torch.rand(N, device=device) >= (1.0 - node_keep_p)) | protect

        # guarantee >= 1 node per graph so batch_size is preserved
        kept_per_graph = torch.zeros(B, device=device)
        kept_per_graph.index_add_(0, batch, keep.float())
        for gid in torch.nonzero(kept_per_graph == 0).view(-1):
            idx = (batch == gid).nonzero(as_tuple=True)[0]
            keep[idx[torch.randint(idx.numel(), (1,), device=device)]] = True

        keep_idx = torch.nonzero(keep).view(-1)
        edge_attr = getattr(data, "edge_attr", None)
        sub_ei, sub_ea, emask = subgraph(
            keep_idx, edge_index,
            edge_attr=edge_attr,
            relabel_nodes=True,
            num_nodes=N,
            return_edge_mask=True,
        )

        # edges fully inside R are protected from dropout
        prot_edge = (protect[edge_index[0]] & protect[edge_index[1]])[emask]

        # symmetric (undirected-consistent) edge dropout
        if sub_ei.numel() > 0:
            u, v = sub_ei[0], sub_ei[1]
            key = torch.minimum(u, v) * (keep_idx.numel() + 1) + torch.maximum(u, v)
            uniq, inv = key.unique(return_inverse=True)
            drop = torch.rand(uniq.numel(), device=device) < edge_drop_p
            edge_keep = (~drop[inv]) | prot_edge
            sub_ei = sub_ei[:, edge_keep]
            if sub_ea is not None:
                sub_ea = sub_ea[edge_keep]

        sub = Data(x=x[keep_idx], edge_index=sub_ei)
        if sub_ea is not None:
            sub.edge_attr = sub_ea
        sub.batch = batch[keep_idx]
        return sub, B

    def forward(self, *args, **kwargs):
        data = kwargs.get("data")
        task = getattr(self.config, "task", None)

        if data is None:
            raise ValueError("GSATGNNs_MODIFIED requires `data` in kwargs.")
        if not hasattr(data, "batch") or data.batch is None:
            data.batch = torch.zeros(data.x.shape[0], dtype=torch.long, device=data.x.device)

        # test inference — plain GNN, ones mask, no game
        if (not self.training) and task == "test":
            logits = self._run_gnn(*args, **kwargs)
            att = torch.ones(data.x.shape[0], device=logits.device)
            att_log = self._to_log_logits(att)
            self._clear_stored()
            return logits, att_log, att_log.sigmoid()

        # warmup — Arthur trains on plain unmasked graphs until he's accurate.
        # Merlin only starts once Arthur has a stable base to search against.
        # Validation during warmup also uses plain GNN so checkpoint selection
        # is based on unmasked accuracy during this phase.
        if self._in_warmup():
            logits = self._run_gnn(*args, **kwargs)
            att = torch.ones(data.x.shape[0], device=logits.device)
            att_log = self._to_log_logits(att)
            self._clear_stored()
            # Deletion-robustness from epoch 0: train Arthur on randomly
            # node/edge-deleted graphs (no mask protected yet). This is the
            # intervention family suff_cause evaluates against.
            if self.training:
                with torch.no_grad():
                    clean_probs = (torch.sigmoid(logits.detach()).view(-1)
                                   if logits.shape[-1] == 1
                                   else logits.detach().softmax(dim=-1))
                protect = torch.zeros(data.x.shape[0], dtype=torch.bool, device=data.x.device)
                sub, Bsz = self._suff_augment(data, protect)
                logits_int = self._run_gnn(data=sub, batch_size=Bsz)
                self.suff_loss = (_graph_loss(logits_int, data.y)
                                  + _consistency_loss(logits_int, clean_probs))
            return logits, att_log, att_log.sigmoid()

        # post-warmup: Merlin always runs (training AND validation AND eval_metric)
        
        emb = self.gnn(*args, without_readout=True, **kwargs)
        emb_det = emb.detach()

        # Clean full-graph prediction through the plain
        if self.training:
            logits_clean = self._run_gnn(*args, **kwargs)
        else:
            with torch.no_grad():
                logits_clean = self._run_gnn(*args, **kwargs)
        with torch.no_grad():
            clean_probs = (torch.sigmoid(logits_clean.detach()).view(-1)
                           if logits_clean.shape[-1] == 1
                           else logits_clean.detach().softmax(dim=-1))

        # Merlin inner loop uses soft mask through full masked-GNN path.
        # Arthur/GNN weights are frozen so gradients flow only into alpha.
        def merlin_fn(node_mask):
            nm = node_mask.unsqueeze(-1) if node_mask.dim() == 1 else node_mask
            # use soft mask directly — better gradient signal than binary
            nm = nm.clamp(0.01, 0.99)
            ea = self._node_to_edge_att(nm, data)
            set_masks(ea, self, nm)
            try:
                return self._run_gnn(*args, **kwargs)
            finally:
                clear_masks(self)

        param_states = [(p, p.requires_grad) for p in self.parameters()]
        for p in self.parameters():
            p.requires_grad_(False)
        try:
            with torch.enable_grad():
                for _ in range(self.game_iterations):
                    soft_mask, binary_mask = self.merlin(
                        embeddings=emb_det,
                        edge_index=data.edge_index,
                        batch=data.batch,
                        classifier_fn=merlin_fn,
                        target=data.y,
                        target_probs=clean_probs,
                    )
        finally:
            for p, req in param_states:
                p.requires_grad_(req)

        # FIX 1: no softening — use identical hard binary mask at train and eval
        # Previously: att = 0.15 + 0.85 * binary_mask (training only)
        # This caused a distribution shift that made masks fail at eval time.
        att = binary_mask.float().clamp(0.05, 0.95)

        edge_att = self._node_to_edge_att(att, data)
        set_masks(edge_att, self, att.unsqueeze(-1))
        logits_merlin = self._run_gnn(*args, **kwargs)
        clear_masks(self)

        self.edge_mask = edge_att
        att_log = self._to_log_logits(att)

        # Morgana and verifier — training only
        if self.training:
            # verifier on Merlin — detached so only verifier_head trains
            emb_merlin_pooled = self._mean_pool(emb.detach() * att.unsqueeze(1), data.batch)
            self.verifier_merlin = self.verifier_head(emb_merlin_pooled)

            # Morgana: PGD attack on detached embeddings
            with torch.enable_grad():
                adv_emb, worst_loss = self.morgana(
                    embeddings=emb_det,
                    edge_index=data.edge_index,
                    batch=data.batch,
                    classifier_fn=self._pool_and_classify,
                    target=data.y,
                )

            self.last_certificate = {
                "L_min": _graph_loss(logits_merlin, data.y).item(),
                "L_max": worst_loss,
            }

            self.logits_morgana = self._pool_and_classify(adv_emb, data.edge_index, data.batch)
            self.att_morgana    = torch.ones_like(att)

            # verifier on Morgana — detached
            emb_morgana_pooled = self._mean_pool(adv_emb.detach(), data.batch)
            self.verifier_morgana = self.verifier_head(emb_morgana_pooled)

            self.morgana_loss       = None
            self.morgana_loss_value = None

        
            self.clean_logits = logits_clean
            protect = binary_mask.bool()
            ce_list, cons_list = [], []
            for _ in range(self.n_suff_aug):
                sub, Bsz = self._suff_augment(data, protect)
                logits_int = self._run_gnn(data=sub, batch_size=Bsz)
                ce_list.append(_graph_loss(logits_int, data.y))
                cons_list.append(_consistency_loss(logits_int, clean_probs))
            self.suff_loss = (torch.stack(ce_list).mean()
                              + torch.stack(cons_list).max())
        else:
            self._clear_stored()

        return logits_merlin, att_log, att_log.sigmoid().clamp(0.05, 0.95)

    @staticmethod
    def concrete_sample(att_log_logit, temp, training):
        if training:
            noise = torch.empty_like(att_log_logit).uniform_(1e-10, 1 - 1e-10)
            noise = torch.log(noise) - torch.log(1.0 - noise)
            return ((att_log_logit + noise) / temp).sigmoid()
        return att_log_logit.sigmoid()

    @torch.no_grad()
    def probs(self, *args, **kwargs):
        logits = self(*args, **kwargs)[0]
        return logits.softmax(dim=-1) if logits.shape[-1] > 1 else logits.sigmoid()

    @torch.no_grad()
    def predict_from_subgraph(self, edge_att=False, log=None, eval_kl=None,
                               node_att=False, edge_attn=None, *args, **kwargs):
        data = kwargs.get("data", None)
        if edge_attn is not None:
            edge_att = edge_attn
        if (data is not None and node_att is not False and node_att is not None
                and (edge_att is False or edge_att is None)):
            if node_att.dim() == 1:
                node_att = node_att.unsqueeze(-1)
            node_att = torch.nan_to_num(node_att.float(), nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
            edge_att = self._node_to_edge_att(node_att, data)

        set_masks(edge_att, self, node_att)
        lc_logits = self.classifierS(self.gnn_clf(*args, **kwargs)) if self.gnn_clf \
                    else self.classifierS(self.gnn(*args, **kwargs))
        clear_masks(self)

        if log is None:
            return lc_logits.softmax(dim=1) if lc_logits.shape[-1] > 1 else lc_logits.sigmoid()
        assert eval_kl is not None
        if lc_logits.shape[-1] > 1:
            return lc_logits.log_softmax(dim=1)
        if eval_kl:
            p = lc_logits.sigmoid()
            out = torch.zeros(p.shape[0], 2, device=p.device)
            out[:, 1] = p.squeeze(1)
            out[:, 0] = 1 - out[:, 1]
            out[out == 0.0] = 1e-10
            return out.log()
        return lc_logits.sigmoid().log()

    def get_subgraph(self, *args, **kwargs):
        orig_task = getattr(self.config, "task", None)
        self.config.task = "eval_metric"
        try:
            logits, att_log, att = self.forward(*args, **kwargs)
        finally:
            self.config.task = orig_task
        scores = att if att is not None else att_log.sigmoid()
        if scores.dim() == 1:
            scores = scores.unsqueeze(-1)
        scores = torch.nan_to_num(scores.float(), nan=0.5, posinf=0.95, neginf=0.05).clamp(0.05, 0.95)
        return scores, scores, logits


def set_masks(mask, model, node_mask=None):
    modules = model.gnn.encoder.convs.modules() if model.gnn_clf is None \
              else model.gnn_clf.encoder.convs.modules()
    if node_mask is not None and node_mask.dim() == 1:
        node_mask = node_mask.unsqueeze(-1)
    for m in modules:
        if isinstance(m, MessagePassing):
            if __pyg_version__ == "2.4.0":
                m._fixed_explain = True
            else:
                m.__explain__ = True
                m._explain    = True
            m._apply_sigmoid = False
            m._edge_mask     = mask
            m._node_mask     = node_mask


def clear_masks(model):
    modules = model.gnn.encoder.convs.modules() if model.gnn_clf is None \
              else model.gnn_clf.encoder.convs.modules()
    for m in modules:
        if isinstance(m, MessagePassing):
            if __pyg_version__ == "2.4.0":
                m._fixed_explain = False
            else:
                m.__explain__ = False
                m._explain    = False
            m._edge_mask = None
            m._node_mask = None


class ExtractorMLP(nn.Module):
    def __init__(self, *args, **kwargs):
        raise RuntimeError("ExtractorMLP removed — use MerlinExtractor / MorganaExtractor.")
    def forward(self, *args, **kwargs):
        raise RuntimeError("ExtractorMLP removed — use MerlinExtractor / MorganaExtractor.")
