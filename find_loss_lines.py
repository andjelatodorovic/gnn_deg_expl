from pathlib import Path

p = Path("GOOD/kernel/pipelines/basic_pipeline.py")
text = p.read_text(encoding="utf-8", errors="replace").splitlines()

keywords = ["spec_loss", "mean_loss", "total_loss", "save_checkpoint", "ckpt"]
for i, line in enumerate(text, start=1):
    if any(k in line for k in keywords):
        print(f"{i}: {line}")
