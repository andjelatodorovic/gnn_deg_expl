from pathlib import Path
import re

p = Path("GOOD/kernel/pipelines/basic_pipeline.py")
text = p.read_text(encoding="utf-8", errors="replace")

# -------------------------------------------------------------------
# Add helper function before BasicPipeline class if not already present
# -------------------------------------------------------------------

helper = r'''
def _mma_graph_loss_for_modified(logits, target):
    """Binary/multiclass-safe Morgana soundness loss for GSATGNNs_MODIFIED."""
    import torch
    import torch.nn.functional as F

    if logits is None:
        return None

    if logits.shape[-1] == 1:
        return F.binary_cross_entropy_with_logits(
            logits.view(-1),
            target.float().view(-1),
        )

    return F.cross_entropy(
        logits,
        target.long().view(-1),
    )


def _maybe_add_morgana_loss(config, model, loss, data):
    """Add Morgana soundness loss only for GSATGNNs_MODIFIED.

    The model still returns the usual 3 GOOD/GSAT outputs.
    Morgana logits are stored internally as model.logits_morgana.
    """
    try:
        model_name = config.model.model_name
    except Exception:
        model_name = None

    if model_name != "GSATGNNs_MODIFIED":
        return loss

    logits_morgana = getattr(model, "logits_morgana", None)
    if logits_morgana is None:
        return loss

    target = getattr(data, "y", None)
    if target is None:
        return loss

    morgana_loss = _mma_graph_loss_for_modified(logits_morgana, target)
    if morgana_loss is None:
        return loss

    try:
        extra_param = config.ood.extra_param
        morgana_weight = float(extra_param[6]) if len(extra_param) > 6 else 1.0
    except Exception:
        morgana_weight = 1.0

    # Store for optional inspection/logging.
    try:
        model.morgana_loss_value = float(morgana_loss.detach().cpu())
    except Exception:
        pass

    return loss + morgana_weight * morgana_loss

'''

if "def _maybe_add_morgana_loss" not in text:
    if "class BasicPipeline" not in text:
        raise RuntimeError("Could not find class BasicPipeline insertion point.")
    text = text.replace("class BasicPipeline", helper + "\nclass BasicPipeline", 1)
    print("Inserted Morgana loss helper functions.")
else:
    print("Morgana loss helper already present.")

# -------------------------------------------------------------------
# Insert just before loss.backward()
# -------------------------------------------------------------------

needle = "loss.backward()"

if "_maybe_add_morgana_loss(self.config, self.model, loss, data)" in text:
    print("Morgana loss call already inserted.")
else:
    idx = text.find(needle)
    if idx == -1:
        raise RuntimeError("Could not find loss.backward() in basic_pipeline.py")

    # Find indentation of loss.backward line.
    line_start = text.rfind("\n", 0, idx) + 1
    line = text[line_start:text.find("\n", idx)]
    indent = line[:len(line) - len(line.lstrip())]

    insert = (
        f"{indent}# Optional Morgana soundness loss for GSATGNNs_MODIFIED.\n"
        f"{indent}loss = _maybe_add_morgana_loss(self.config, self.model, loss, data)\n"
    )

    text = text[:line_start] + insert + text[line_start:]
    print("Inserted Morgana loss call before loss.backward().")

p.write_text(text, encoding="utf-8")
print("Patched basic_pipeline.py with optional Morgana loss.")
