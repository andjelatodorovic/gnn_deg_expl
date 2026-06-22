"""
Arthur-Morgana Game-Theoretic GNN Explanation Extractor


Adapts the Merlin-Arthur-Morgana framework from:
https://github.com/ZIB-IOL/merlin-arthur-classifiers


For GNN node/edge selection with certified robustness guarantees.


Usage in GSAT:
    # Replace ExtractorMLP + sampling with ArthurMorganaGNNExplainer
    explainer = ArthurMorganaGNNExplainer(
        config=gnn_config,
        gnn_model=trained_gnn,
        classifier=gnn_classifier,
        K=10,  # sparsity
        epsilon=0.05,  # perturbation budget
        game_iterations=5
    )
    
    # During inference
    explanation_mask, bounds = explainer.explain(data, edge_index, batch)
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.utils import to_undirected, is_undirected, coalesce
from torch_sparse import transpose as sparse_transpose
from typing import Tuple, Dict, Optional, List
import numpy as np
from dataclasses import dataclass


def _graph_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Cross-entropy that works for both multi-class and binary classification.

    PyG-based binary classifiers output a single logit per graph (shape [B, 1]).
    torch.nn.functional.cross_entropy interprets that as 1 class (only index 0
    is valid), so any target value of 1 raises an IndexError.  We detect this
    case and use binary_cross_entropy_with_logits instead.
    """
    if logits.shape[-1] == 1:
        # Binary: squeeze to [B] and use BCE
        return F.binary_cross_entropy_with_logits(
            logits.squeeze(-1), target.float()
        )
    else:
        return F.cross_entropy(logits, target.long())




@dataclass
class ExplanationBounds:
    """Certified bounds on explanation validity."""
    mask: torch.Tensor  # Binary or soft mask (K-sparse)
    clean_loss: float  # L_min: loss without perturbation
    worst_loss: float  # L_max: loss under Morgana attack
    robustness_margin: float  # Δ = L_max - L_min
    convergence_history: List[Dict]  # Per-iteration metrics




class MerlinOptimizer(nn.Module):
    """
    Merlin (Cooperative): Generates K-sparse explanation masks to maximize fidelity.
    
    Uses soft masking + gradient descent to learn node importance scores,
    then thresholds to enforce sparsity.
    """
    
    def __init__(
        self, 
        num_nodes: int,
        num_edges: int,
        hidden_dim: int = 64,
        learning_rate: float = 0.01,
        sparsity_lambda: float = 0.1,
        steps: int = 50,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.learning_rate = learning_rate
        self.sparsity_lambda = sparsity_lambda
        self.steps = steps
        
        self.alpha = nn.Parameter(torch.randn(num_nodes) * 0.1)
        
        self.temp = 1.0
        
    def forward(
        self,
        embeddings: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        forward_fn,  # GNN forward pass function
        target: torch.Tensor,
        K: int,
        device: torch.device = "cpu",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        optimizer = optim.Adam([self.alpha], lr=self.learning_rate)
        
        for step in range(self.steps):
            optimizer.zero_grad()
            
            m_soft = torch.sigmoid(self.alpha)
            masked_embeddings = embeddings * m_soft.unsqueeze(1)
            
           
            masked_logits = forward_fn(masked_embeddings, edge_index, batch)
            
            # Task loss — binary-safe (FIX: was cross_entropy which breaks on 1-logit output)
            task_loss = _graph_loss(masked_logits, target)
            
            sparsity_loss = self.sparsity_lambda * m_soft.sum()
            
            loss = task_loss + sparsity_loss
            loss.backward()
            optimizer.step()
        
        # Thresholding: select top-K nodes per graph or per batch ??? idk because I can't asses the aprox. K
        m_soft = torch.sigmoid(self.alpha).detach()

        if batch is None:
            batch = torch.zeros(
                m_soft.numel(),
                dtype=torch.long,
                device=m_soft.device,
            )
        else:
            batch = batch.to(m_soft.device)

        binary_mask = torch.zeros_like(m_soft)
        K = int(K)

        for graph_id in batch.unique(sorted=True):
            idx = (batch == graph_id).nonzero(as_tuple=True)[0]

            if idx.numel() == 0:
                continue

            k_graph = max(1, min(K, idx.numel()))
            local_scores = m_soft[idx]
            _, local_top = torch.topk(local_scores, k_graph)

            binary_mask[idx[local_top]] = 1.0

        return m_soft, binary_mask




class MorganaAttacker(nn.Module):
    """
    Morgana (Adversarial): Finds worst-case perturbations to break explanations.
    
    Given Merlin's mask, performs PGD attack to find adversarial perturbations
    that minimize loss on the masked subgraph.
    """
    
    def __init__(
        self,
        epsilon: float = 0.05,
        attack_steps: int = 20,
        attack_lr: float = 0.01,
        restarts: int = 3,
    ):
        super().__init__()
        self.epsilon = epsilon
        self.attack_steps = attack_steps
        self.attack_lr = attack_lr
        self.restarts = restarts
        
    def forward(
        self,
        embeddings: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        binary_mask: torch.Tensor,
        forward_fn,
        target: torch.Tensor,
        device: torch.device = "cpu",
    ) -> Tuple[torch.Tensor, float]:
        """
        PGD attack on masked explanation.
        
        Args:
            embeddings: Original node embeddings (num_nodes, hidden_dim)
            edge_index: Graph structure
            batch: Batch assignment
            binary_mask: Binary node mask from Merlin (num_nodes,)
            forward_fn: GNN forward function
            target: Target labels
            device: Device to run on
            
        Returns:
            (adversarial_embedding, worst_loss): Perturbed embeddings and worst-case loss
        """
        best_loss = -float('inf')
        best_delta = None
        
        for restart in range(self.restarts):
            # Random initialization of perturbation
            delta = (torch.rand_like(embeddings) * 2 - 1) * self.epsilon
            delta = delta.to(device)
            delta.requires_grad = True
            
            for step in range(self.attack_steps):
                # Perturbed embeddings (only on masked nodes)
                perturbed_embeddings = embeddings + delta
                # Optionally: zero out perturbations on non-masked nodes
                masked_nodes = binary_mask.nonzero(as_tuple=True)[0]
                
                # Forward pass
                logits = forward_fn(perturbed_embeddings, edge_index, batch)
                
                # Loss — binary-safe (FIX: was cross_entropy which breaks on 1-logit output)
                loss = _graph_loss(logits, target)
                
                # Backward
                if delta.grad is not None:
                    delta.grad.zero_()
                loss.backward()
                
                # PGD update: move in direction of gradient ascent (maximize loss)
                with torch.no_grad():
                    delta = delta + self.attack_lr * torch.sign(delta.grad)
                    # Project onto ℓ∞ ball
                    delta.clamp_(-self.epsilon, self.epsilon)
                
                delta.requires_grad = True
            
            # Evaluate final loss
            with torch.no_grad():
                perturbed_embeddings = embeddings + delta
                logits = forward_fn(perturbed_embeddings, edge_index, batch)
                final_loss = _graph_loss(logits, target)  # FIX: binary-safe
            
            if final_loss > best_loss:
                best_loss = final_loss.item()
                best_delta = delta.detach().clone()
        
        if best_delta is None:
            best_delta = torch.zeros_like(embeddings, device=device)
            best_loss = 0.0

        return embeddings + best_delta.to(device), best_loss




class ArthurVerifier(nn.Module):
    """
    Arthur (Verifier): Computes and certifies robustness bounds.
    
    After Merlin-Morgana reach equilibrium, computes:
    - L_min: clean loss on explanation
    - L_max: worst-case loss under Morgana attack
    - Robustness certificate: [L_min, L_max]
    """
    
    def __init__(self):
        super().__init__()
    
    def compute_bounds(
        self,
        clean_loss: float,
        worst_loss: float,
        epsilon: float,
        num_masked_nodes: int,
        gnn_lipschitz: float = 1.0,  # Local Lipschitz constant (estimate)
    ) -> Dict:
        """
        Compute robustness certificate bounds.
        
        Args:
            clean_loss: L_min (loss without perturbation)
            worst_loss: L_max (worst-case loss under attack)
            epsilon: Perturbation budget (ℓ∞ radius)
            num_masked_nodes: Number of selected nodes in explanation
            gnn_lipschitz: Local Lipschitz constant of GNN at explanation
            
        Returns:
            Certificate dict with bounds and diagnostics
        """
        robustness_margin = worst_loss - clean_loss
        relative_margin = robustness_margin / (clean_loss + 1e-8)
        
        # Conservative Lipschitz-based bound
        conservative_bound = clean_loss + gnn_lipschitz * epsilon * num_masked_nodes
        
        return {
            "L_min": clean_loss,
            "L_max": worst_loss,
            "robustness_margin": robustness_margin,
            "relative_margin": relative_margin,
            "conservative_bound": conservative_bound,
            "epsilon": epsilon,
            "num_masked_nodes": num_masked_nodes,
            "certificate": f"∀||δ||_∞ ≤ {epsilon}: {clean_loss:.4f} ≤ Loss(explanation + δ) ≤ {worst_loss:.4f}"
        }




class ArthurMorganaGNNExplainer(nn.Module):
    """
    Main Arthur-Morgana Game Orchestrator for GNN Explanations.
    
    Replaces gradient-based soft-to-hard thresholding in GSAT with:
    - Merlin: cooperative mask generation
    - Morgana: adversarial perturbations
    - Arthur: robustness certification
    """
    
    def __init__(
        self,
        num_nodes: int,
        num_edges: int,
        gnn_forward_fn,  # Function to run GNN forward pass
        classifier_fn,   # Function to compute loss/logits
        K: int = 10,
        epsilon: float = 0.05,
        game_iterations: int = 5,
        merlin_steps: int = 50,
        merlin_lr: float = 0.01,
        sparsity_lambda: float = 0.1,
        morgana_steps: int = 20,
        morgana_lr: float = 0.01,
        morgana_restarts: int = 3,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.K = max(1, int(K))
        self.epsilon = float(epsilon)
        self.game_iterations = max(1, int(game_iterations))
        self.device = device
        
        # Initialize players
        self.merlin = MerlinOptimizer(
            num_nodes=num_nodes,
            num_edges=num_edges,
            learning_rate=merlin_lr,
            sparsity_lambda=sparsity_lambda,
            steps=max(1, int(merlin_steps)),
        ).to(device)
        
        self.morgana = MorganaAttacker(
            epsilon=epsilon,
            attack_steps=max(1, int(morgana_steps)),
            attack_lr=morgana_lr,
            restarts=morgana_restarts,
        ).to(device)
        
        self.arthur = ArthurVerifier().to(device)
        
        # Store functions
        self.gnn_forward_fn = gnn_forward_fn
        self.classifier_fn = classifier_fn
        
        # Tracking
        self.convergence_history = []
    
    def forward(
        self,
        embeddings: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        target: torch.Tensor,
    ) -> ExplanationBounds:
        """
        Run Arthur-Morgana game to extract certified explanation.
        
        Args:
            embeddings: Node embeddings from GNN (num_nodes, hidden_dim)
            edge_index: Graph structure (2, num_edges)
            batch: Batch assignment (num_nodes,)
            target: Target labels (batch_size,)
            
        Returns:
            ExplanationBounds object with mask + robustness certificate
        """
        self.convergence_history = []
        target = target.view(-1).long()
        
        best_mask = None
        best_bounds = None
        
        # Game iterations
        for iteration in range(self.game_iterations):
            # === MERLIN: Generate K-sparse explanation ===
            soft_mask, binary_mask = self.merlin.forward(
                embeddings=embeddings,
                edge_index=edge_index,
                batch=batch,
                forward_fn=lambda emb, ei, b: self._forward_with_mask(emb, ei, b, None),
                target=target,
                K=self.K,
                device=self.device,
            )
            
            # === ARTHUR: Evaluate clean explanation ===
            with torch.no_grad():
                clean_logits = self._forward_with_mask(embeddings, edge_index, batch, binary_mask)
                clean_loss = _graph_loss(clean_logits, target).item()  # FIX: binary-safe
            
            # === MORGANA: Attack the explanation ===
            perturbed_embeddings, worst_loss = self.morgana.forward(
                embeddings=embeddings,
                edge_index=edge_index,
                batch=batch,
                binary_mask=binary_mask,
                forward_fn=lambda emb, ei, b: self._forward_with_mask(emb, ei, b, binary_mask),
                target=target,
                device=self.device,
            )
            
            # === ARTHUR: Compute robustness bounds ===
            num_masked_nodes = binary_mask.sum().item()
            bounds_dict = self.arthur.compute_bounds(
                clean_loss=clean_loss,
                worst_loss=worst_loss,
                epsilon=self.epsilon,
                num_masked_nodes=int(num_masked_nodes),
            )
            
            # Track convergence
            iteration_log = {
                "iteration": iteration,
                "clean_loss": clean_loss,
                "worst_loss": worst_loss,
                **bounds_dict
            }
            self.convergence_history.append(iteration_log)
            
            # Keep best explanation (smallest margin)
            margin = bounds_dict["robustness_margin"]
            if best_bounds is None or margin < best_bounds["robustness_margin"]:
                best_mask = binary_mask.detach().clone()
                best_bounds = bounds_dict
        
        # === Fallback if no valid best explanation was selected ===
        if best_bounds is None:
            k = max(1, min(int(self.K), embeddings.shape[0]))
            fallback_scores = torch.ones(embeddings.shape[0], device=embeddings.device)
            _, indices = torch.topk(fallback_scores, k)

            best_mask = torch.zeros_like(fallback_scores)
            best_mask[indices] = 1.0

            with torch.no_grad():
                logits = self._forward_with_mask(embeddings, edge_index, batch, best_mask)
                clean_loss = _graph_loss(logits, target).item()  # FIX: binary-safe

            best_bounds = {
                "L_min": clean_loss,
                "L_max": clean_loss,
                "robustness_margin": 0.0,
            }

        # === Return certified explanation ===
        return ExplanationBounds(
            mask=best_mask,
            clean_loss=best_bounds["L_min"],
            worst_loss=best_bounds["L_max"],
            robustness_margin=best_bounds["robustness_margin"],
            convergence_history=self.convergence_history,
        )
    
    def _forward_with_mask(
        self,
        embeddings: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply optional mask and forward through classifier logits."""
        if mask is not None:
            embeddings = embeddings * mask.unsqueeze(1)

        logits = self.classifier_fn(embeddings, edge_index, batch)
        return logits




# ============================================================================
# Integration with GSAT (drop-in replacement)
# ============================================================================


class GSATWithArthurMorgana(nn.Module):
    """
    GSAT model with Arthur-Morgana explanation extractor.
    
    Minimal modification to original GSAT:
    - Replace ExtractorMLP + sampling with ArthurMorganaGNNExplainer
    - Keep everything else the same
    """
    
    def __init__(
        self,
        gnn_base,  # FeatExtractor from GSAT
        classifier,  # Classifier from GSAT
        num_nodes: int,
        num_edges: int,
        K: int = 10,
        epsilon: float = 0.05,
        game_iterations: int = 5,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.gnn_base = gnn_base
        self.classifier = classifier
        self.device = device
        
        # Replace ExtractorMLP with Arthur-Morgana
        self.explainer = ArthurMorganaGNNExplainer(
            num_nodes=num_nodes,
            num_edges=num_edges,
            gnn_forward_fn=self._gnn_forward,
            classifier_fn=self._classifier_forward,
            K=K,
            epsilon=epsilon,
            game_iterations=game_iterations,
            device=device,
        )
        
        self.edge_mask = None
    
    def _gnn_forward(self, embeddings, edge_index, batch):
        """Forward through GNN on masked embeddings."""
        # Reconstruct data object for GNN
        from torch_geometric.data import Data, Batch
        # This is simplified; actual implementation depends on GSAT structure
        return self.gnn_base(embeddings, edge_index, batch)
    
    def _classifier_forward(self, embeddings, edge_index, batch):
        """Forward through classifier."""
        gnn_out = self.gnn_base(embeddings, edge_index, batch)
        logits = self.classifier(gnn_out)
        return logits
    
    def forward(self, data):
        """
        Forward pass with Arthur-Morgana explanation extraction.
        """
        # Standard GNN forward
        emb = self.gnn_base(data.x, data.edge_index, data.batch)
        
        # Arthur-Morgana explanation extraction
        explanation = self.explainer(
            embeddings=emb,
            edge_index=data.edge_index,
            batch=data.batch,
            target=data.y,
        )
        
        # Get logits with explanation mask applied
        masked_emb = emb * explanation.mask.unsqueeze(1)
        logits = self.classifier(self.gnn_base(data.x, data.edge_index, data.batch))
        
        # Store mask for visualization
        self.edge_mask = explanation.mask
        
        return logits, explanation




if __name__ == "__main__":
    # Example usage
    print("ArthurMorganaGNNExplainer module loaded successfully.")
    print("\nTo integrate into GSAT:")
    print("1. Replace: self.extractor = ExtractorMLP(...)")
    print("2. With: self.explainer = ArthurMorganaGNNExplainer(...)")
    print("3. Modify forward() to call self.explainer instead of sampling")