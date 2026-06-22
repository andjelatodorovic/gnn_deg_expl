from pathlib import Path

p = Path("GOOD/networks/models/GSATGNNs_MODIFIED.py")
text = p.read_text(encoding="utf-8", errors="replace")

method = '''    def _soften_training_attention(self, att: Tensor, floor: float = 0.2) -> Tensor:
        """Soft residual version of Arthur-Morgana's hard mask.

        During training, a fully hard 0/1 mask can starve the classifier.
        This maps:
            0 -> floor
            1 -> 1

        Example with floor=0.2:
            unselected nodes keep 20% signal,
            selected nodes keep 100% signal.
        """
        att = att.float()
        return floor + (1.0 - floor) * att

'''

if "def _soften_training_attention" not in text:
    marker = "    def _attention_to_log_logits"
    if marker not in text:
        raise RuntimeError("Could not find insertion point before _attention_to_log_logits.")
    text = text.replace(marker, method + marker, 1)

p.write_text(text, encoding="utf-8")
print("Inserted _soften_training_attention into GSATGNNs_MODIFIED.py")
