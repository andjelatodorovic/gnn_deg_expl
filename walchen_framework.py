import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, List, Any, Optional
import tqdm


class ArthurClassifier(nn.Module):
    """
    Verifier / Arthur.

    Takes fixed-dimensional Merlin or Morgana feature vectors:
        [batch_size, input_dim]

    Returns binary class logits:
        [batch_size, 2]
    """

    def __init__(
        self,
        num_features: Optional[int] = None,
        input_dim: Optional[int] = None,
        hidden_dim: int = 64,
        num_classes: int = 2,
    ):
        super().__init__()

        if input_dim is None and num_features is None:
            raise ValueError("ArthurClassifier needs either input_dim or num_features.")

        if input_dim is None:
            input_dim = num_features

        self.input_dim = int(input_dim)

        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.dim() == 1:
            features = features.unsqueeze(1)

        return self.net(features.float())


class MerlinExplainer:
    """
    Prover / Merlin.

    For GSAT, tries to use model.get_subgraph(...) first.
    If that fails, falls back to model output.
    """

    def __init__(self, se_gnn_model, device="cpu"):
        self.model = se_gnn_model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    def _extract_tensor(self, obj: Any) -> torch.Tensor:
        """
        Extract the most explanation-like tensor from model output.

        We prefer non-scalar tensors, because a scalar logit is not a useful
        explanation vector.
        """

        tensors = []

        def collect(x):
            if isinstance(x, torch.Tensor):
                tensors.append(x)
            elif isinstance(x, dict):
                for v in x.values():
                    collect(v)
            elif isinstance(x, (list, tuple)):
                for v in x:
                    collect(v)

        collect(obj)

        if not tensors:
            raise ValueError("No tensor found in explanation output.")

        non_scalar = [t for t in tensors if t.numel() > 1]

        if non_scalar:
            vectors = [t for t in non_scalar if t.dim() == 1]
            if vectors:
                chosen = max(vectors, key=lambda t: t.numel())
            else:
                chosen = max(non_scalar, key=lambda t: t.numel())
        else:
            chosen = tensors[0]

        return chosen.detach().float().view(-1).to(self.device)

    def get_explanation(self, graph_data) -> torch.Tensor:
        graph_data = graph_data.to(self.device)

        with torch.no_grad():
            if hasattr(self.model, "get_subgraph"):
                try:
                    out = self.model.get_subgraph(graph_data)
                    return self._extract_tensor(out)
                except Exception as exc:
                    print(
                        f"Warning: get_subgraph failed, falling back to model output: {exc}"
                    )

            out = self.model(graph_data)

            if hasattr(self.model, "explanation"):
                return self._extract_tensor(self.model.explanation)

            if hasattr(self.model, "att_weights"):
                return self._extract_tensor(self.model.att_weights)

            return self._extract_tensor(out)


class MorganaAdversary:
    """
    Adversary / Morgana.

    Selects alternative features from the full graph feature pool while keeping
    the same dimensionality Arthur expects.
    """

    def __init__(self, arthur: ArthurClassifier, device="cpu"):
        self.arthur = arthur
        self.device = torch.device(device)
        self.arthur.to(self.device)
        self.arthur.eval()

    @staticmethod
    def _pad_or_trim(vec: torch.Tensor, target_dim: int) -> torch.Tensor:
        vec = vec.detach().float().view(-1)

        if vec.numel() >= target_dim:
            return vec[:target_dim]

        pad = vec.new_zeros(target_dim - vec.numel())
        return torch.cat([vec, pad], dim=0)

    def generate_attack(
        self,
        full_features: torch.Tensor,
        true_labels: torch.Tensor,
        target_class: Optional[int] = None,
        num_features_to_select: int = 5,
        method: str = "greedy",
    ) -> torch.Tensor:

        full_features = full_features.to(self.device).float()

        if full_features.dim() == 1:
            full_features = full_features.unsqueeze(0)

        true_labels = true_labels.to(self.device).long().view(-1)

        if method == "greedy":
            return self._greedy_attack(
                full_features,
                true_labels,
                target_class,
                num_features_to_select,
            )

        raise ValueError(f"Unknown attack method: {method}")

    def _greedy_attack(
        self,
        full_features: torch.Tensor,
        true_labels: torch.Tensor,
        target_class: Optional[int],
        num_features: int,
    ) -> torch.Tensor:

        batch_size, total_features = full_features.shape

        if total_features <= num_features:
            return torch.stack(
                [
                    self._pad_or_trim(full_features[i], num_features)
                    for i in range(batch_size)
                ],
                dim=0,
            ).to(self.device)

        selected_indices = np.random.choice(
            total_features,
            size=num_features,
            replace=False,
        )

        for _ in range(10):
            selected_features = full_features[:, selected_indices]

            with torch.no_grad():
                logits = self.arthur(selected_features)
                preds = logits.argmax(dim=1)

            current_fool_rate = (preds != true_labels).float().mean().item()

            unselected = [i for i in range(total_features) if i not in selected_indices]

            best_swap = None
            best_fool_rate = current_fool_rate

            for selected_idx in selected_indices:
                for unselected_idx in unselected:
                    temp_indices = selected_indices.copy()
                    swap_pos = np.where(selected_indices == selected_idx)[0][0]
                    temp_indices[swap_pos] = unselected_idx

                    temp_features = full_features[:, temp_indices]

                    with torch.no_grad():
                        temp_logits = self.arthur(temp_features)
                        temp_preds = temp_logits.argmax(dim=1)

                    temp_fool_rate = (temp_preds != true_labels).float().mean().item()

                    if temp_fool_rate > best_fool_rate:
                        best_fool_rate = temp_fool_rate
                        best_swap = (swap_pos, unselected_idx)

            if best_swap is None:
                break

            swap_pos, new_idx = best_swap
            selected_indices[swap_pos] = new_idx

        return full_features[:, selected_indices].to(self.device)


class WalchenFramework:
    """
    Merlin-Arthur verification framework.

    Important correction:
    Arthur is trained on one subset and evaluated on a held-out subset.
    This avoids the earlier problem where shuffled-label controls still passed.
    """

    def __init__(
        self,
        se_gnn_model,
        device="cpu",
        arthur_train_fraction: float = 0.7,
        alpha_cap: float = 10.0,
        random_seed: int = 42,
    ):
        self.device = torch.device(device)
        self.merlin = MerlinExplainer(se_gnn_model, self.device)
        self.arthur_train_fraction = arthur_train_fraction
        self.alpha_cap = alpha_cap
        self.random_seed = random_seed

    @staticmethod
    def _clean_labels(labels: torch.Tensor) -> torch.Tensor:
        return labels.detach().view(-1).long()

    @staticmethod
    def _pad_or_trim(vec: torch.Tensor, target_dim: int) -> torch.Tensor:
        vec = vec.detach().float().view(-1)

        if vec.numel() >= target_dim:
            return vec[:target_dim]

        pad = vec.new_zeros(target_dim - vec.numel())
        return torch.cat([vec, pad], dim=0)

    def _make_train_eval_split(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make a simple random held-out split for Arthur.

        For small samples, keeps at least one example in each subset.
        """

        if n < 2:
            raise ValueError("Need at least 2 graphs for held-out Arthur evaluation.")

        generator = torch.Generator()
        generator.manual_seed(self.random_seed)

        perm = torch.randperm(n, generator=generator)

        n_train = int(round(self.arthur_train_fraction * n))
        n_train = max(1, min(n_train, n - 1))

        train_idx = perm[:n_train].to(self.device)
        eval_idx = perm[n_train:].to(self.device)

        return train_idx, eval_idx

    def run_verification(
        self,
        graphs: List,
        labels: torch.Tensor,
        num_selected_features: int = 5,
    ) -> Dict:

        labels = self._clean_labels(labels).to(self.device)

        print("Step 1: Collecting Merlin's explanations...")
        merlin_explanations = []

        for graph in tqdm.tqdm(graphs):
            explanation = self.merlin.get_explanation(graph)
            merlin_explanations.append(explanation)

        merlin_features = self._explanations_to_features(
            merlin_explanations,
            num_selected_features,
        )

        if merlin_features.dim() == 1:
            merlin_features = merlin_features.unsqueeze(1)

        merlin_features = merlin_features.float().to(self.device)
        actual_feature_dim = merlin_features.shape[1]

        print(f"\nMerlin feature tensor shape: {tuple(merlin_features.shape)}")
        print(f"Arthur input dim set to: {actual_feature_dim}")

        train_idx, eval_idx = self._make_train_eval_split(len(labels))

        print(f"Arthur train graphs: {train_idx.numel()}")
        print(f"Arthur eval graphs:  {eval_idx.numel()}")

        merlin_train = merlin_features[train_idx]
        labels_train = labels[train_idx]

        merlin_eval = merlin_features[eval_idx]
        labels_eval = labels[eval_idx]

        print("\nStep 2: Training Arthur (Verifier) on train subset...")
        arthur = ArthurClassifier(input_dim=actual_feature_dim).to(self.device)

        self._fit_arthur(
            arthur=arthur,
            features=merlin_train,
            labels=labels_train,
        )

        print("\nStep 3: Evaluating completeness on held-out subset...")
        completeness_error = self._classification_error(
            arthur=arthur,
            features=merlin_eval,
            labels=labels_eval,
        )

        print("\nStep 4: Generating Morgana's adversarial features on held-out subset...")
        morgana = MorganaAdversary(arthur, self.device)

        eval_graphs = [graphs[int(i.detach().cpu().item())] for i in eval_idx]

        morgana_features = self._get_morgana_features(
            eval_graphs,
            labels_eval,
            morgana,
            actual_feature_dim,
        )

        if morgana_features.dim() == 1:
            morgana_features = morgana_features.unsqueeze(1)

        morgana_features = morgana_features.float().to(self.device)

        print(f"Morgana feature tensor shape: {tuple(morgana_features.shape)}")

        print("\nStep 5: Testing Arthur against Morgana on held-out subset...")
        soundness_error = self._test_soundness(
            arthur,
            morgana_features,
            labels_eval,
        )

        print("\nStep 6: Computing mutual information bounds...")
        afc, alpha_raw, alpha_used = self._estimate_afc_and_alpha(
            merlin_eval,
            morgana_features,
            labels_eval,
        )

        mutual_info_bound = self._compute_mutual_info_bound(
            completeness_error,
            soundness_error,
            afc,
            alpha_used,
        )

        return {
            "completeness": completeness_error,
            "soundness": soundness_error,
            "mutual_info_bound": mutual_info_bound,
            "afc": afc,
            "alpha": alpha_used,
            "alpha_raw": alpha_raw,
            "alpha_cap": self.alpha_cap,
            "merlin_features": merlin_features.detach().cpu(),
            "morgana_features": morgana_features.detach().cpu(),
            "arthur_model": arthur,
            "arthur_train_size": int(train_idx.numel()),
            "arthur_eval_size": int(eval_idx.numel()),
            "arthur_train_indices": train_idx.detach().cpu(),
            "arthur_eval_indices": eval_idx.detach().cpu(),
        }

    def _explanations_to_features(
        self,
        explanations: List[torch.Tensor],
        num_features: int,
    ) -> torch.Tensor:

        features_list = []

        for expl in explanations:
            expl = expl.detach().float().view(-1).to(self.device)

            if expl.numel() == 0:
                top_k = torch.zeros(num_features, device=self.device)
            else:
                scores = expl.abs()
                k = min(num_features, scores.numel())
                top_k = torch.topk(scores, k, largest=True).values

                if k < num_features:
                    pad = torch.zeros(num_features - k, device=self.device)
                    top_k = torch.cat([top_k, pad], dim=0)

            features_list.append(top_k)

        return torch.stack(features_list, dim=0).to(self.device)

    def _fit_arthur(
        self,
        arthur: ArthurClassifier,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:

        optimizer = torch.optim.Adam(arthur.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()

        features = features.to(self.device).float()
        labels = self._clean_labels(labels).to(self.device)

        arthur.train()

        for _ in range(75):
            optimizer.zero_grad()
            logits = arthur(features)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

    def _classification_error(
        self,
        arthur: ArthurClassifier,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> float:

        arthur.eval()

        features = features.to(self.device).float()
        labels = self._clean_labels(labels).to(self.device)

        with torch.no_grad():
            logits = arthur(features)
            preds = logits.argmax(dim=1)
            error = (preds != labels).float().mean().item()

        return float(error)

    def _graph_to_candidate_features(self, graph) -> torch.Tensor:
        graph = graph.to(self.device)

        if hasattr(graph, "x") and graph.x is not None:
            features = graph.x.detach().float().view(-1)
        elif hasattr(graph, "edge_attr") and graph.edge_attr is not None:
            features = graph.edge_attr.detach().float().view(-1)
        elif hasattr(graph, "edge_index") and graph.edge_index is not None:
            features = graph.edge_index.detach().float().view(-1)
        else:
            features = torch.ones(1, device=self.device)

        if features.numel() == 0:
            features = torch.ones(1, device=self.device)

        return features

    def _get_morgana_features(
        self,
        graphs: List,
        labels: torch.Tensor,
        morgana: MorganaAdversary,
        num_features: int,
    ) -> torch.Tensor:

        labels = self._clean_labels(labels).to(self.device)
        morgana_features_list = []

        for i, graph in enumerate(tqdm.tqdm(graphs)):
            candidate_features = self._graph_to_candidate_features(graph)
            candidate_features = candidate_features.unsqueeze(0)

            attack_feat = morgana.generate_attack(
                full_features=candidate_features,
                true_labels=labels[i : i + 1],
                num_features_to_select=num_features,
                method="greedy",
            )

            attack_feat = attack_feat.squeeze(0)
            attack_feat = self._pad_or_trim(attack_feat, num_features)

            morgana_features_list.append(attack_feat)

        return torch.stack(morgana_features_list, dim=0).to(self.device)

    def _test_soundness(
        self,
        arthur: ArthurClassifier,
        morgana_features: torch.Tensor,
        labels: torch.Tensor,
    ) -> float:

        arthur.eval()

        morgana_features = morgana_features.to(self.device).float()
        labels = self._clean_labels(labels).to(self.device)

        with torch.no_grad():
            logits = arthur(morgana_features)
            preds = logits.argmax(dim=1)
            error = (preds != labels).float().mean().item()

        return float(error)

    def _estimate_afc_and_alpha(
        self,
        merlin_features: torch.Tensor,
        morgana_features: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[float, float, float]:

        merlin_features = merlin_features.to(self.device).float()
        morgana_features = morgana_features.to(self.device).float()
        labels = self._clean_labels(labels).to(self.device)

        class_0_mask = labels == 0
        class_1_mask = labels == 1

        if class_0_mask.sum() > 0 and class_1_mask.sum() > 0:
            feat_0_mean = merlin_features[class_0_mask].mean(dim=0)
            feat_1_mean = merlin_features[class_1_mask].mean(dim=0)

            denom = feat_0_mean.norm() * feat_1_mean.norm()

            if denom.item() > 0:
                corr = torch.cosine_similarity(
                    feat_0_mean.unsqueeze(0),
                    feat_1_mean.unsqueeze(0),
                    dim=1,
                ).item()
                kappa = max(1.0, abs(float(corr)))
            else:
                kappa = 1.0
        else:
            kappa = 1.0

        diff = (merlin_features - morgana_features).abs().mean().item()
        scale = merlin_features.abs().mean().item() + 1e-8

        alpha_raw = max(1e-6, diff / scale)

        # Important: prevent alpha explosions from creating false PASS results.
        alpha_used = min(alpha_raw, self.alpha_cap)

        print(f"AFC kappa:        {kappa:.4f}")
        print(f"Alpha raw:        {alpha_raw:.4f}")
        print(f"Alpha used/capped:{alpha_used:.4f}")

        return float(kappa), float(alpha_raw), float(alpha_used)

    def _compute_mutual_info_bound(
        self,
        epsilon_c: float,
        epsilon_s: float,
        kappa: float,
        alpha: float,
    ) -> float:

        epsilon_c = float(epsilon_c)
        epsilon_s = float(epsilon_s)
        kappa = float(kappa)
        alpha = max(float(alpha), 1e-8)

        if epsilon_c >= 1.0:
            return 0.0

        B = 1.0

        numerator = kappa * (1.0 / alpha) * epsilon_s
        denominator = 1.0 - epsilon_c + (
            kappa * (1.0 / (alpha * B)) * epsilon_s
        )

        if denominator <= 0:
            return 0.0

        bound = 1.0 - epsilon_c - numerator / denominator

        return float(max(0.0, bound))


if __name__ == "__main__":
    pass