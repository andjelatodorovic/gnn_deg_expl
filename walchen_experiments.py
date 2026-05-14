"""
walchen_experiments.py

Run Wälchen verification on a trained GOOD / GSAT model.

Place this file in the repo root:

    C:/Users/Cerebria/gnn_deg_expl/walchen_experiments.py

Required sibling file:

    C:/Users/Cerebria/gnn_deg_expl/walchen_framework.py

Example from VS Code terminal:

    cd C:/Users/Cerebria/gnn_deg_expl
    C:/Users/Cerebria/.virtualenv/Scripts/activate

    python walchen_experiments.py `
      --config_path final_configs/BAColorGVIsol/basis/no_shift/GSAT.yaml `
      --seed 1 `
      --pretrain degenerate `
      --backbone ACR2 `
      --split id_test `
      --load_split id `
      --device cpu `
      --num-features 5 `
      --limit-graphs 50 `
      --output ./walchen_experiments/
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch_geometric.data import Batch, Data


# ---------------------------------------------------------------------
# Repo-root setup
# ---------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_DISABLED", "true")


# ---------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------

from walchen_framework import WalchenFramework

from GOOD import config_summoner
from GOOD.utils.args import args_parser
from GOOD.utils.loader import initialize_model_dataset
from GOOD.ood_algorithms.ood_manager import load_ood_alg
from GOOD.kernel.pipeline_manager import load_pipeline


# ---------------------------------------------------------------------
# JSON / tensor helpers
# ---------------------------------------------------------------------

def to_jsonable(obj: Any) -> Any:
    """Convert tensors, numpy values and nested objects into JSON-safe values."""
    if isinstance(obj, torch.Tensor):
        obj = obj.detach().cpu()
        if obj.numel() == 1:
            return obj.item()
        return obj.tolist()

    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]

    if isinstance(obj, tuple):
        return [to_jsonable(v) for v in obj]

    try:
        import numpy as np

        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
    except Exception:
        pass

    return obj


def get_graph_label(graph: Data) -> torch.Tensor:
    """Extract a single graph-level label from a PyG Data object."""
    y = graph.y

    if not isinstance(y, torch.Tensor):
        return torch.tensor(int(y), dtype=torch.long)

    y = y.detach().cpu()

    if y.numel() == 1:
        return y.view(-1)[0].long()

    return y.view(-1)[0].long()


def maybe_limit_graphs(
    graphs: List[Data],
    labels: torch.Tensor,
    limit_graphs: Optional[int],
) -> Tuple[List[Data], torch.Tensor]:
    """Optionally restrict verification to the first N graphs."""
    if limit_graphs is None or limit_graphs <= 0:
        return graphs, labels

    limit = min(limit_graphs, len(graphs))
    return graphs[:limit], labels[:limit]


# ---------------------------------------------------------------------
# GOOD / GSAT adapter
# ---------------------------------------------------------------------

class GOODModelAdapter(torch.nn.Module):
    """
    Adapter around the trained GOOD / GSAT model.

    The repo normally calls the model as:

        model(
            data=data,
            edge_weight=None,
            ood_algorithm=ood_algorithm,
            ...
        )

    Wälchen is likely to expect something simpler like:

        model(graphs)

    This wrapper bridges that interface.
    """

    def __init__(self, model: torch.nn.Module, ood_algorithm: Any, config: Any):
        super().__init__()
        self.model = model
        self.ood_algorithm = ood_algorithm
        self.config = config
        self.device = config.device

    def _to_batch(self, graph_or_batch: Any) -> Batch:
        """Convert Data, Batch or list[Data] into a PyG Batch."""
        if isinstance(graph_or_batch, list):
            data = Batch.from_data_list(graph_or_batch)
        elif isinstance(graph_or_batch, Data):
            data = Batch.from_data_list([graph_or_batch])
        else:
            data = graph_or_batch

        return data.to(self.device)

    def _raw_forward(self, graph_or_batch: Any) -> Any:
        """Call the underlying GOOD model in the repo's expected style."""
        data = self._to_batch(graph_or_batch)

        self.model.eval()

        with torch.no_grad():
            model_output = self.model(
                data=data,
                edge_weight=None,
                ood_algorithm=self.ood_algorithm,
                max_num_epoch=getattr(self.config.train, "max_epoch", None),
                curr_epoch=getattr(self.config.train, "epoch", 0),
                pretrain=False,
            )

        return model_output

    def forward(self, graph_or_batch: Any) -> torch.Tensor:
        """
        Return the model's prediction logits after OOD postprocessing.

        For GSAT, the raw model output is a tuple:
            logits, attention_logits, attention

        and GSAT.output_postprocess(...) returns logits.
        """
        model_output = self._raw_forward(graph_or_batch)

        if hasattr(self.ood_algorithm, "output_postprocess"):
            output = self.ood_algorithm.output_postprocess(model_output)
        else:
            output = model_output

        if isinstance(output, (tuple, list)):
            output = output[0]

        return output

    @torch.no_grad()
    def predict(self, graph_or_batch: Any) -> torch.Tensor:
        """Return hard class predictions."""
        output = self.forward(graph_or_batch)

        if output.ndim == 2 and output.size(-1) > 1:
            return output.argmax(dim=-1)

        return (output.view(-1) > 0).long()

    @torch.no_grad()
    def predict_proba(self, graph_or_batch: Any) -> torch.Tensor:
        """Return probabilities when possible."""
        output = self.forward(graph_or_batch)

        if output.ndim == 2 and output.size(-1) > 1:
            return torch.softmax(output, dim=-1)

        return torch.sigmoid(output.view(-1))

    @torch.no_grad()
    def get_subgraph(self, graph_or_batch: Any) -> Any:
        """
        Expose GSAT's subgraph/explanation method.

        GSAT in this repo implements get_subgraph(...), returning:
            edge_mask, attention, logits
        """
        data = self._to_batch(graph_or_batch)

        if not hasattr(self.model, "get_subgraph"):
            raise AttributeError(
                "Underlying GOOD model has no get_subgraph(...) method."
            )

        return self.model.get_subgraph(
            data=data,
            edge_weight=None,
            ood_algorithm=self.ood_algorithm,
            do_relabel=False,
        )


# ---------------------------------------------------------------------
# GOOD / GSAT loading
# ---------------------------------------------------------------------

def build_good_args(
    config_path: str,
    seed: int,
    pretrain: str,
    backbone: str,
    device: str,
) -> Any:
    """
    Build repo-compatible arguments.

    Must match the training run. For your trained run, this should be:

        --seed 1
        --pretrain degenerate
        --backbone ACR2
    """

    gpu_idx = "0" if device == "cuda" else "-1"

    argv = [
        "--config_path", config_path,
        "--task", "test",
        "--seeds", str(seed),
        "--exp_round", str(seed),
        "--random_seed", str(seed),
        "--pretrain", pretrain,
        "--backbone", backbone,
        "--wandb", "False",
        "--gpu_idx", gpu_idx,
    ]

    return args_parser(argv)


def load_model_and_data(
    config_path: str,
    seed: int,
    pretrain: str,
    backbone: str,
    split: str,
    load_split: str,
    device: str,
    checkpoint: Optional[str] = None,
    limit_graphs: Optional[int] = None,
) -> Tuple[GOODModelAdapter, List[Data], torch.Tensor, Dict[str, Any]]:
    """
    Load trained GOOD / GSAT model and selected dataset split.

    Args:
        config_path:
            Path relative to configs/, e.g.
            final_configs/BAColorGVIsol/basis/no_shift/GSAT.yaml

        seed:
            Experiment round / seed used during training.

        pretrain:
            Must match training, e.g. degenerate.

        backbone:
            Must match training, e.g. ACR2.

        split:
            Which dataset split to verify, e.g. id_test or test.

        load_split:
            Which checkpoint to load:
                id  -> id_best.ckpt
                ood -> best.ckpt

        device:
            cpu or cuda.

        checkpoint:
            Optional explicit checkpoint path.

        limit_graphs:
            Optional first-N subset for a quicker first run.
    """

    print("\n" + "=" * 60)
    print("LOADING TRAINED GOOD / GSAT MODEL")
    print("=" * 60)
    print(f"Config path: {config_path}")
    print(f"Seed/round:  {seed}")
    print(f"Pretrain:    {pretrain}")
    print(f"Backbone:    {backbone}")
    print(f"Split:       {split}")
    print(f"Load split:  {load_split}")
    print(f"Device:      {device}")

    good_args = build_good_args(
        config_path=config_path,
        seed=seed,
        pretrain=pretrain,
        backbone=backbone,
        device=device,
    )

    config = config_summoner(good_args)

    if device == "cpu":
        config.device = torch.device("cpu")
    elif device == "cuda":
        if torch.cuda.is_available():
            config.device = torch.device("cuda:0")
        else:
            print("CUDA requested but unavailable. Falling back to CPU.")
            config.device = torch.device("cpu")
    else:
        raise ValueError(f"Unsupported device: {device}")

    print("\nResolved configuration:")
    print(f"  Dataset:       {config.dataset.dataset_name}")
    print(f"  Model:         {config.model.model_name}")
    print(f"  OOD algorithm: {config.ood.ood_alg}")
    print(f"  Device:        {config.device}")
    print(f"  Checkpoint dir:{config.ckpt_dir}")
    print(f"  OOD ckpt:      {config.test_ckpt}")
    print(f"  ID ckpt:       {config.id_test_ckpt}")

    model, loader = initialize_model_dataset(config)
    ood_algorithm = load_ood_alg(config.ood.ood_alg, config)

    pipeline = load_pipeline(
        config.pipeline,
        config.task,
        model,
        loader,
        ood_algorithm,
        config,
    )

    model.to(config.device)

    if checkpoint is not None:
        checkpoint_path = Path(checkpoint).resolve()

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Explicit checkpoint not found: {checkpoint_path}")

        print(f"\nLoading explicit checkpoint:\n  {checkpoint_path}")

        ckpt = torch.load(str(checkpoint_path), map_location=config.device)

        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            model.load_state_dict(ckpt["state_dict"])
        elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt)

    else:
        print("\nLoading checkpoint using the repo pipeline...")
        print(f"  load_split={load_split}")

        # Same loading mechanism as the repo's test flow.
        pipeline.load_task(load_param=True, load_split=load_split)

    model.eval()

    if not isinstance(loader, dict):
        raise TypeError(
            "Expected loader to be a dict of splits. "
            f"Got loader type: {type(loader)}"
        )

    if split not in loader:
        raise KeyError(
            f"Split '{split}' not available. Available splits: {list(loader.keys())}"
        )

    dataset = loader[split].dataset

    graphs = [dataset[i].cpu() for i in range(len(dataset))]
    labels = torch.stack([get_graph_label(graph) for graph in graphs]).long()

    graphs, labels = maybe_limit_graphs(graphs, labels, limit_graphs)

    adapted_model = GOODModelAdapter(model, ood_algorithm, config).to(config.device)
    adapted_model.eval()

    dataset_info = {
        "config_path": config_path,
        "resolved_dataset_name": config.dataset.dataset_name,
        "resolved_model_name": config.model.model_name,
        "resolved_ood_alg": config.ood.ood_alg,
        "seed": seed,
        "pretrain": pretrain,
        "backbone": backbone,
        "split": split,
        "load_split": load_split,
        "device": str(config.device),
        "ckpt_dir": str(config.ckpt_dir),
        "test_ckpt": str(config.test_ckpt),
        "id_test_ckpt": str(config.id_test_ckpt),
        "explicit_checkpoint": str(checkpoint) if checkpoint else None,
        "num_graphs": len(graphs),
        "num_labels": int(labels.numel()),
        "num_classes": getattr(config.dataset, "num_classes", None),
    }

    return adapted_model, graphs, labels, dataset_info


# ---------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------

def sanity_check_loaded_model(
    model: GOODModelAdapter,
    graphs: List[Data],
    labels: torch.Tensor,
    max_graphs: int = 8,
) -> None:
    """
    Check that loading worked before running Wälchen.

    This verifies:
      1. graphs exist
      2. labels exist
      3. the model forward pass works
      4. outputs are finite
      5. prediction count matches graph count
    """

    print("\n" + "=" * 60)
    print("SANITY CHECK")
    print("=" * 60)

    if len(graphs) == 0:
        raise ValueError("No graphs loaded.")

    if len(labels) != len(graphs):
        raise ValueError(
            f"Label/graph mismatch: {len(labels)} labels for {len(graphs)} graphs."
        )

    small_graphs = graphs[: min(max_graphs, len(graphs))]
    small_labels = labels[: len(small_graphs)]

    print(f"Loaded graphs: {len(graphs)}")
    print(f"Loaded labels: {len(labels)}")
    print(f"Checking first {len(small_graphs)} graphs")

    for i, graph in enumerate(small_graphs[:3]):
        num_edges = (
            graph.edge_index.size(1)
            if hasattr(graph, "edge_index") and graph.edge_index is not None
            else None
        )

        x_shape = tuple(graph.x.shape) if hasattr(graph, "x") and graph.x is not None else None

        print(
            f"Graph {i}: "
            f"num_nodes={graph.num_nodes}, "
            f"num_edges={num_edges}, "
            f"x_shape={x_shape}, "
            f"y={graph.y}"
        )

    with torch.no_grad():
        output = model(small_graphs)

    print(f"\nModel output shape: {tuple(output.shape)}")
    print(f"Model output sample:\n{output[:3]}")

    if not torch.isfinite(output).all():
        raise ValueError("Model output contains NaN or Inf.")

    preds = model.predict(small_graphs)

    print(f"\nPredictions: {preds.detach().cpu().view(-1).tolist()}")
    print(f"Labels:      {small_labels.detach().cpu().view(-1).tolist()}")

    if preds.view(-1).numel() != len(small_graphs):
        raise ValueError(
            f"Prediction count mismatch: got {preds.view(-1).numel()}, "
            f"expected {len(small_graphs)}."
        )

    print("\nSanity check passed.")


# ---------------------------------------------------------------------
# Wälchen
# ---------------------------------------------------------------------

def run_walchen_verification(
    model: GOODModelAdapter,
    graphs: List[Data],
    labels: torch.Tensor,
    device: str,
    num_features: int,
) -> Dict[str, Any]:
    print("\n" + "=" * 60)
    print("RUNNING WÄLCHEN VERIFICATION")
    print("=" * 60)
    print(f"Graphs:        {len(graphs)}")
    print(f"Labels:        {len(labels)}")
    print(f"Device:        {device}")
    print(f"Num features:  {num_features}\n")

    framework = WalchenFramework(model, device=device)

    results = framework.run_verification(
        graphs=graphs,
        labels=labels,
        num_selected_features=num_features,
    )

    return results


def print_results(results: Dict[str, Any], dataset_info: Dict[str, Any]) -> str:
    print("\n" + "=" * 60)
    print("WÄLCHEN VERIFICATION RESULTS")
    print("=" * 60)
    print(f"Model:   {dataset_info.get('resolved_model_name')}")
    print(f"Dataset: {dataset_info.get('resolved_dataset_name')}")
    print(f"Split:   {dataset_info.get('split')}")
    print(f"Graphs:  {dataset_info.get('num_graphs')}\n")

    completeness = float(to_jsonable(results.get("completeness", float("nan"))))
    soundness = float(to_jsonable(results.get("soundness", float("nan"))))
    afc = float(to_jsonable(results.get("afc", float("nan"))))
    alpha = float(to_jsonable(results.get("alpha", float("nan"))))
    mutual_info = float(to_jsonable(results.get("mutual_info_bound", float("nan"))))

    print(f"Completeness (epsilon_c):     {completeness:.4f}")
    print("  Lower is better.")

    print(f"\nSoundness (epsilon_s):        {soundness:.4f}")
    print("  Higher is better.")

    print(f"\nAsymmetric Feature Corr kappa:{afc:.4f}")
    print("  Measures feature bias in the dataset.")

    print(f"\nRelative Success Rate alpha:  {alpha:.4f}")
    print("  Measures Morgana attack effectiveness.")

    print("\n" + "-" * 60)
    print("MUTUAL INFORMATION GUARANTEE:")
    print(f"I(Features; Class) >= {mutual_info:.4f}")
    print("-" * 60)

    if mutual_info > 0.5:
        status = "PASS - strong mutual information guarantee"
    elif mutual_info > 0.0:
        status = "PARTIAL - weak mutual information guarantee"
    else:
        status = "FAIL - no mutual information guarantee"

    print(f"\nStatus: {status}\n")

    return status


def save_results(
    results: Dict[str, Any],
    dataset_info: Dict[str, Any],
    status: str,
    output_dir: str,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    save_dict = {
        "status": status,
        "dataset_info": to_jsonable(dataset_info),
        "results": to_jsonable(results),
    }

    output_file = output_path / "walchen_results.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(save_dict, f, indent=2)

    print(f"Results saved to: {output_file}")

    return output_file


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Wälchen verification on a trained GOOD / GSAT model."
    )

    parser.add_argument(
        "--config_path",
        "--config",
        dest="config_path",
        type=str,
        default="final_configs/BAColorGVIsol/basis/no_shift/GSAT.yaml",
        help="Path relative to configs/, or absolute path.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Experiment round / seed used during training.",
    )

    parser.add_argument(
        "--pretrain",
        type=str,
        default="degenerate",
        help="Must match the training run.",
    )

    parser.add_argument(
        "--backbone",
        type=str,
        default="ACR2",
        help="Must match the training run.",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="id_test",
        choices=["train", "eval_train", "id_val", "id_test", "val", "test"],
        help="Dataset split to pass to Wälchen.",
    )

    parser.add_argument(
        "--load_split",
        type=str,
        default="id",
        choices=["id", "ood"],
        help="Which checkpoint to load: id_best.ckpt or best.ckpt.",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional explicit checkpoint path. If omitted, repo checkpoint logic is used.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Use cpu first on Windows unless CUDA is definitely working.",
    )

    parser.add_argument(
        "--num-features",
        type=int,
        default=5,
        help="Number of features Wälchen should select.",
    )

    parser.add_argument(
        "--limit-graphs",
        type=int,
        default=50,
        help="Use first N graphs for a quicker first run. Use 0 for all graphs.",
    )

    parser.add_argument(
        "--skip-sanity-check",
        action="store_true",
        help="Skip the pre-Wälchen forward-pass sanity check.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="./walchen_experiments/",
        help="Output directory for JSON results.",
    )

    return parser.parse_args()


def save_results(
    results: Dict[str, Any],
    dataset_info: Dict[str, Any],
    status: str,
    output_dir: str,
) -> Path:
    """
    Save Wälchen results.

    JSON cannot serialize PyTorch modules, so Arthur is saved separately
    as arthur_model.pt.
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save Arthur separately if present.
    arthur_model = results.get("arthur_model", None)
    arthur_path = None

    if arthur_model is not None:
        arthur_path = output_path / "arthur_model.pt"
        torch.save(arthur_model.state_dict(), arthur_path)
        print(f"Arthur model saved to: {arthur_path}")

    # Remove non-JSON-serializable objects.
    results_for_json = {
        key: value
        for key, value in results.items()
        if key != "arthur_model"
    }

    save_dict = {
        "status": status,
        "dataset_info": to_jsonable(dataset_info),
        "results": to_jsonable(results_for_json),
        "arthur_model_path": str(arthur_path) if arthur_path else None,
    }

    output_file = output_path / "walchen_results.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(save_dict, f, indent=2)

    print(f"Results saved to: {output_file}")

    return output_file


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Wälchen verification on a trained GOOD / GSAT model."
    )

    parser.add_argument(
        "--config_path",
        "--config",
        dest="config_path",
        type=str,
        default="final_configs/BAColorGVIsol/basis/no_shift/GSAT.yaml",
        help="Path relative to configs/, or absolute path.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Experiment round / seed used during training.",
    )

    parser.add_argument(
        "--pretrain",
        type=str,
        default="degenerate",
        help="Must match the training run.",
    )

    parser.add_argument(
        "--backbone",
        type=str,
        default="ACR2",
        help="Must match the training run.",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="id_test",
        choices=["train", "eval_train", "id_val", "id_test", "val", "test"],
        help="Dataset split to pass to Wälchen.",
    )

    parser.add_argument(
        "--load_split",
        type=str,
        default="id",
        choices=["id", "ood"],
        help="Which checkpoint to load: id_best.ckpt or best.ckpt.",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional explicit checkpoint path. If omitted, repo checkpoint logic is used.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Use cpu first on Windows unless CUDA is definitely working.",
    )

    parser.add_argument(
        "--num-features",
        type=int,
        default=5,
        help="Number of features Wälchen should select.",
    )

    parser.add_argument(
        "--limit-graphs",
        type=int,
        default=50,
        help="Use first N graphs for a quicker first run. Use 0 for all graphs.",
    )

    parser.add_argument(
        "--skip-sanity-check",
        action="store_true",
        help="Skip the pre-Wälchen forward-pass sanity check.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="./walchen_experiments/",
        help="Output directory for JSON results.",
    )

    parser.add_argument(
        "--shuffle-labels",
        action="store_true",
        help="Negative control: randomly shuffle labels before Wälchen verification.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    limit_graphs = None if args.limit_graphs == 0 else args.limit_graphs

    model, graphs, labels, dataset_info = load_model_and_data(
        config_path=args.config_path,
        seed=args.seed,
        pretrain=args.pretrain,
        backbone=args.backbone,
        split=args.split,
        load_split=args.load_split,
        device=args.device,
        checkpoint=args.checkpoint,
        limit_graphs=limit_graphs,
    )

    if not args.skip_sanity_check:
        sanity_check_loaded_model(model, graphs, labels)

    if args.shuffle_labels:
        perm = torch.randperm(labels.size(0))
        labels = labels[perm]
        print("\nWARNING: Running shuffled-label negative control.")
        print("Labels have been randomly permuted.\n")

    results = run_walchen_verification(
        model=model,
        graphs=graphs,
        labels=labels,
        device=str(model.device),
        num_features=args.num_features,
    )

    status = print_results(results, dataset_info)

    save_results(
        results=results,
        dataset_info=dataset_info,
        status=status,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
