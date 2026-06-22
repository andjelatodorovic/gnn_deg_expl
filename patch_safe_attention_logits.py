from pathlib import Path
import re

p = Path("GOOD/networks/models/GSATGNNs_MODIFIED.py")
text = p.read_text(encoding="utf-8", errors="replace")

new_helper = '''    def _attention_to_log_logits(self, att: Tensor) -> Tensor:
        """Convert attention mask/probabilities into safe GSAT-style logits.

        GOOD/GSAT losses may apply entropy or information regularisation to
        sigmoid(att_log_logits). Therefore we must avoid exact 0 or 1 values.
        """
        att = att.float()

        if att.dim() == 1:
            att = att.unsqueeze(-1)

        eps = 1e-4
        att = att.clamp(eps, 1.0 - eps)

        return torch.logit(att)
'''

pattern = r"    def _attention_to_log_logits\(self, att: Tensor\) -> Tensor:\n.*?\n(?=    # ------------------------------------------------------------------\n    # Forward)"

text, n = re.subn(pattern, new_helper + "\n", text, flags=re.DOTALL)

if n == 0:
    raise RuntimeError("Could not find _attention_to_log_logits block.")

p.write_text(text, encoding="utf-8")
print("Patched _attention_to_log_logits with safe clamp.")
