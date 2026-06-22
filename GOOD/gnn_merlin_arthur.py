"""


Arthur-Morgana GNN Explanation Extractor

Adapts the Merlin-Arthur-Morgana framework from:
https://github.com/ZIB-IOL/merlin-arthur-classifiers


to use in GSAT:
    # Replace ExtractorMLP + sampling 
    
    w/ the following code
    explainer = ArthurMorganaGNNExplainer(
        config=gnn_config,
        gnn_model=trained_gnn,
        classifier=gnn_classifier,
        K=10,  # sparsity
        epsilon=0.05,  # perturbation budget
        game_iterations=5
    )
    
    
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.utils import to_undirected, is_undirected, transpose, coalesce
from torch_sparse import transpose as sparse_transpose
from typing import Tuple, Dict, Optional, List
import numpy as np
from dataclasses import dataclass


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
        
        # Soft mask parameterization: α ∈ ℝ^{num_nodes}
        self.alpha = nn.Parameter(torch.randn(num_nodes) * 0.1)
        
        # Optional: learnable temperature for concrete distribution
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
            
            # Soft mask via sigmoid
            m_soft = torch.sigmoid(self.alpha)
            
            masked_embeddings = embeddings * m_soft.unsqueeze(1)
            
            # Forward pass on masked graph (idk if forward_fn wil lhandle embeddings well )
            masked_logits = forward_fn(masked_embeddings, edge_index, batch)
            
            task_loss = torch.nn.functional.cross_entropy(masked_logits, target)
            sparsity_loss = self.sparsity_lambda * m_soft.sum()   #encourage masks to be sparse
            
          
            loss = task_loss + sparsity_loss
            loss.backward()
            optimizer.step()
        
        # Thresholding top-K nodes
        m_soft = torch.sigmoid(self.alpha).detach()
        _, indices = torch.topk(m_soft, K)
        binary_mask = torch.zeros_like(m_soft)
        binary_mask[indices] = 1.0
        
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
        

        # Masking explanations then performing a pdg attacl
        best_loss = float('inf')
        best_delta = None
        
        for restart in range(self.restarts):
            # Random initialization of perturbation
            delta = (torch.rand_like(embeddings) * 2 - 1) * self.epsilon
            delta = delta.to(device)
            delta.requires_grad = True
            
            for step in range(self.attack_steps):
                # Perturbed embeddings (only on masked nodes)
                perturbed_embeddings = embeddings + delta
                # Optionally: zero out perturbations on non-masked nodes ?? but left to test 
                masked_nodes = binary_mask.nonzero(as_tuple=True)[0]
                
                
                logits = forward_fn(perturbed_embeddings, edge_index, batch)
               
                loss = torch.nn.functional.cross_entropy(logits, target)
                
                # Backward
                if delta.grad is not None:
                    delta.grad.zero_()
                loss.backward()
                
                # PGD update
                with torch.no_grad():
                    delta = delta - self.attack_lr * torch.sign(delta.grad)
                    # clamp/project onto ℓ∞ ball
                    delta.clamp_(-self.epsilon, self.epsilon)
                
                delta.requires_grad = True
            
            #
            with torch.no_grad():
                perturbed_embeddings = embeddings + delta
                logits = forward_fn(perturbed_embeddings, edge_index, batch)
                final_loss = torch.nn.functional.cross_entropy(logits, target)
            
            if final_loss < best_loss:
                best_loss = final_loss.item()
                best_delta = delta.detach().clone()
        
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
   
    
    def __init__(
        self,
        num_nodes: int,
        num_edges: int,
        gnn_forward_fn,  
        classifier_fn,   
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
        self.K = K
        self.epsilon = epsilon
        self.game_iterations = game_iterations
        self.device = device
        
        # Initialize players
        self.merlin = MerlinOptimizer(
            num_nodes=num_nodes,
            num_edges=num_edges,
            learning_rate=merlin_lr,
            sparsity_lambda=sparsity_lambda,
            steps=merlin_steps,
        ).to(device)
        
        self.morgana = MorganaAttacker(
            epsilon=epsilon,
            attack_steps=morgana_steps,
            attack_lr=morgana_lr,
            restarts=morgana_restarts,
        ).to(device)
        
        self.arthur = ArthurVerifier().to(device)
        
        self.gnn_forward_fn = gnn_forward_fn
        self.classifier_fn = classifier_fn
        
        self.convergence_history = []
    
    def forward(
        self,
        embeddings: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        target: torch.Tensor,
    ) -> ExplanationBounds:
       
        self.convergence_history = []
        
        best_mask = None
        best_bounds = None
        
        # Game iterations hyppr
        for iteration in range(self.game_iterations):
            # Merlin
            soft_mask, binary_mask = self.merlin.forward(
                embeddings=embeddings,
                edge_index=edge_index,
                batch=batch,
                forward_fn=lambda emb, ei, b: self._forward_with_mask(emb, ei, b, binary_mask),
                target=target,
                K=self.K,
                device=self.device,
            )
            
            # Here Arthur to evaluate clean explanation ===
            with torch.no_grad():
                masked_embeddings = embeddings * binary_mask.unsqueeze(1)
                clean_logits = self._forward_with_mask(masked_embeddings, edge_index, batch, binary_mask)
                clean_loss = torch.nn.functional.cross_entropy(clean_logits, target).item()
            
            # Morgana attack
            perturbed_embeddings, worst_loss = self.morgana.forward(
                embeddings=embeddings,
                edge_index=edge_index,
                batch=batch,
                binary_mask=binary_mask,
                forward_fn=lambda emb, ei, b: self._forward_with_mask(emb, ei, b, binary_mask),
                target=target,
                device=self.device,
            )
            
            # Arthur robustness bounds ===
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
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Helper to apply mask and forward through GNN."""
        # Apply mask to embeddings
        masked_embeddings = embeddings * mask.unsqueeze(1)
        
        # Forward through GNN
        logits = self.gnn_forward_fn(masked_embeddings, edge_index, batch)
        
        return logits


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
        from torch_geometric.data import Data, Batch  # This is simplified
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
        #attack merlin and take the exact merlin table 2 with column with our approach

        emb = self.gnn_base(data.x, data.edge_index, data.batch)
        

        #separate what happens in the runtime vs the training time // expl. extractor is only arthur

        #add metrics on sufficiency/ necessity
       
        explanation = self.explainer(
            embeddings=emb,
            edge_index=data.edge_index,
            batch=data.batch,
            target=data.y,
        )
        
        masked_emb = emb * explanation.mask.unsqueeze(1)
        logits = self.classifier(self.gnn_base(data.x, data.edge_index, data.batch))
        
        self.edge_mask = explanation.mask
        
        return logits, explanation


if __name__ == "__main__":
    pass