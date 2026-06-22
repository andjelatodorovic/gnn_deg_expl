from pathlib import Path
import re

p = Path("GOOD/networks/models/GSATGNNs_MODIFIED.py")
text = p.read_text(encoding="utf-8", errors="replace")

new_helper = '''    def _attention_to_log_logits(self, att: Tensor) -> Tensor:
        """Convert attention probabilities into safe GSAT-style logits.

        The GSAT auxiliary/spec loss is numerically fragile if attention is
        exactly 0/1, nan, inf, or too saturated. This helper guarantees a
        finite [num_nodes, 1] logit tensor.
        """
        att = att.float()

        if att.dim() == 1:
            att = att.unsqueeze(-1)

        # Remove numerical garbage before logit.
        att = torch.nan_to_num(att, nan=0.5, posinf=0.999, neginf=0.001)

        # Keep away from exact 0/1. Use a slightly wider clamp than 1e-6.
        eps = 1e-3
        att = att.clamp(eps, 1.0 - eps)

        logits = torch.logit(att)

        # Final safety.
        logits = torch.nan_to_num(logits, nan=0.0, posinf=6.0, neginf=-6.0)
        logits = logits.clamp(-6.0, 6.0)

        return logits
'''

pattern = r"    def _attention_to_log_logits\(self, att: Tensor\) -> Tensor:\n.*?\n(?=    # ------------------------------------------------------------------\n    # Forward)"

text, n = re.subn(pattern, new_helper + "\n", text, flags=re.DOTALL)

if n != 1:
    raise RuntimeError(f"Expected to replace one _attention_to_log_logits, replaced {n}")

# Add safety before the final AM return. This targets the block where
# node_att_for_logits and att_log_logits are created.
old = '''        att_log_logits = self._attention_to_log_logits(node_att_for_logits)

        return logits, att_log_logits, node_att_for_logits
'''

new = '''        node_att_for_logits = torch.nan_to_num(
            node_att_for_logits.float(),
            nan=0.5,
            posinf=0.999,
            neginf=0.001,
        ).clamp(1e-3, 1.0 - 1e-3)

        att_log_logits = self._attention_to_log_logits(node_att_for_logits)

        # Keep returned attention finite and probability-shaped.
        returned_att = att_log_logits.sigmoid().clamp(1e-3, 1.0 - 1e-3)

        return logits, att_log_logits, returned_att
'''

if old not in text:
    print("Warning: final AM return block not found exactly.")
else:
    text = text.replace(old, new, 1)

p.write_text(text, encoding="utf-8")
print("Patched safe finite attention logits/return values.")
