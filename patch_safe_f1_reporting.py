from pathlib import Path

p = Path("GOOD/kernel/main.py")
text = p.read_text(encoding="utf-8", errors="replace")

old = '''    if not np.isnan(test_f1_pos["train"][0]):'''

new = '''    if (
        "train" in test_f1_pos
        and len(test_f1_pos["train"]) > 0
        and not np.isnan(test_f1_pos["train"][0])
    ):'''

if old not in text:
    raise RuntimeError("Could not find the unsafe test_f1_pos indexing line.")

text = text.replace(old, new, 1)

p.write_text(text, encoding="utf-8")
print("Patched GOOD/kernel/main.py safe test_f1_pos indexing.")
